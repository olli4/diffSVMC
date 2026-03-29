MODULE water_mod

!--------------------------------------------------------------
! References:
! Launiainen, S., Guan, M., Salmivaara, A., Kieloaho, A.-J., 2019. 
! Modeling boreal forest evapotranspiration and water balance at stand and catchment scales: a spatial approach. 
! Hydrology and Earth System Sciences 23, 3457–3480. https://doi.org/10.5194/hess-23-3457-2019
!--------------------------------------------------------------

! Modules
  use readctrl_mod       ! module for reading control parameters
  use readvegpara_mod    ! module for reading vegetation parameters
  use readsoilpara_mod   ! module for reading soil properties (shared with yasso)

  implicit none

  !Public member functions:
  public :: initialization_spafhy
  public :: canopy_water_flux   ! require inputdata for p-hydro module

  public :: soil_water
  public :: soil_water_retention_curve
  public :: soil_hydraulic_conductivity

  private :: aerodynamics        ! aerodynamic conductances
  private :: canopy_water_snow   ! canopy interception, evaporation and snowpack
  private :: ground_evaporation  ! ground evaporation from top soil layer

  private :: penman_monteith      !
  private :: e_sat                !


contains

  SUBROUTINE initialization_spafhy(canopywater_state, soilwater_state, spafhy_para)
    ! initializes canopy and soil water balance model
    type(soilwater_state_type), intent(inout)    :: soilwater_state
    type(canopywater_state_type), intent(inout)  :: canopywater_state
    type(spafhy_para_type), intent(in)    :: spafhy_para
    real(8)   :: psis, khydr

    soilwater_state%MaxWatSto = 1000 * spafhy_para%soil_depth * spafhy_para%max_poros ! mm
    soilwater_state%FcSto = 1000 * spafhy_para%soil_depth * spafhy_para%fc ! mm, storage at field capacity
    soilwater_state%MaxPondSto = spafhy_para%maxpond  ! mm

    ! initial state
    soilwater_state%WatSto  = 0.9 * soilwater_state%MaxWatSto ! 90% saturation
    soilwater_state%PondSto = 0.0 ! no ponding
    soilwater_state%Wliq =  spafhy_para%max_poros * min(1.0, (soilwater_state%WatSto / soilwater_state%MaxWatSto))
    soilwater_state%Sat  = soilwater_state%Wliq / spafhy_para%max_poros
    soilwater_state%beta = min(1.0, soilwater_state%Wliq / spafhy_para%max_poros) ! modifier for soil evaporation [-]
    
    call soil_water_retention_curve(soilwater_state%Wliq, spafhy_para, psis) ! soil water potential [MPa]
    soilwater_state%Psi = psis

    call soil_hydraulic_conductivity(soilwater_state%Wliq, spafhy_para, khydr) ! hydraulic conductivity [m s-1]
    soilwater_state%Kh = khydr

    ! canopywater
    canopywater_state%CanopyStorage=0.0

    canopywater_state%SWE=0.0
    canopywater_state%swe_i=0.0
    canopywater_state%swe_l=0.0

  END SUBROUTINE initialization_spafhy

  SUBROUTINE reset_spafhy_flux(canopywater_flux, soilwater_flux)
    type(soilwater_flux_type), intent(inout)    :: soilwater_flux
    type(canopywater_flux_type), intent(inout)  :: canopywater_flux
        
    ! soilwater flux
    soilwater_flux%Infiltration = 0.0
    soilwater_flux%Runoff = 0.0
    soilwater_flux%Drainage = 0.0
    soilwater_flux%LateralFlow = 0.0
    soilwater_flux%ET = 0.0
    soilwater_flux%mbe = 0.0

    ! canopywater flux
    canopywater_flux%Throughfall=0.0
    canopywater_flux%Interception=0.0
    canopywater_flux%CanopyEvap=0.0
    canopywater_flux%Unloading=0.0
    canopywater_flux%SoilEvap=0.0
    canopywater_flux%ET=0.0
    canopywater_flux%Transpiration=0.0
    canopywater_flux%PotInfiltration=0.0
    canopywater_flux%Melt=0.0
    canopywater_flux%Freeze=0.0
    canopywater_flux%mbe=0.0

  END SUBROUTINE reset_spafhy_flux
  
  SUBROUTINE soil_water(soilwater_state, soilwater_flux, spafhy_para, potinf, tr, evap, latflow)
  ! ---------------------------------------------------------------
  ! Soil water balance in 1-layer bucket. Pond storage can exist above top layer.   
  ! Updates state, return fluxes
  ! ----------------------------------------------------------------

  ! !ARGUMENTS:
    type(soilwater_state_type), intent(inout)    :: soilwater_state    
    type(soilwater_flux_type), intent(inout)    :: soilwater_flux    
    type(spafhy_para_type), intent(in)    :: spafhy_para
    real(8)      , intent(in)    :: potinf  ! potinf = potential infiltration [mm = kg m-2]    
    real(8)      , intent(inout)    :: tr      ! transpiration from root zone [mm]
    real(8)      , intent(inout)    :: evap    ! evaporation from top layer [mm]
    real(8)      , intent(inout)    :: latflow ! latflow - lateral flow (for ditch drainage) [mm = kg m-2]

  ! ! Local variables 
    real(8)     ::    rr, PondSto0, WatSto0, PondSto, WatSto
    real(8)     ::    drain=0.0, infil=0.0, to_pond=0.0, runoff=0.0
    real(8)     ::    eps=1e-16, Krate=0.0, psis, khydr

    ! storages are in [mm] and input fluxes in [mm during timestep]

    latflow = 0.0

    ! old state at beg. of timestep
    WatSto0 = soilwater_state%WatSto
    PondSto0 = soilwater_state%PondSto

    WatSto = WatSto0
    PondSto = PondSto0

    ! potential infiltration to soil layer
    rr = potinf + soilwater_state%PondSto
    PondSto = 0.0
    
    ! transpiration
    tr = min(tr, WatSto + rr - eps)
    
    ! evaporation 
    evap = min(evap, WatSto + rr - tr - eps)

    ! update waterstorage after upward fluxes
    WatSto = WatSto0 - tr - evap

    ! vertical drainage can happen until field capacity
    drain = soilwater_state%Kh * (time_step * 3600 * 1000) ! mm
    !drain = min(drain, max(0.0, WatSto - eps))
    drain = min(drain, max(0.0, (WatSto - soilwater_state%FcSto)))

    ! lateral drainage to ditches here!
    latflow = 0.0
    latflow = max(0.0, min(latflow, WatSto - drain - eps))

    ! infiltration is limited by available water (rr) or storage space
    infil = min(rr, soilwater_state%MaxWatSto + drain + latflow)
    
    ! update soil water storage
    WatSto = WatSto + infil - drain - latflow

    ! update pond storage
    if (infil < rr) then
      to_pond = rr - infil ! mm
      PondSto = min(PondSto + to_pond, soilwater_state%MaxPondSto) 
      runoff =  max(0.0, to_pond - PondSto)
    end if

    ! update soilwater_flux
    soilwater_flux%Infiltration = infil
    soilwater_flux%Drainage = drain
    soilwater_flux%ET = tr + evap
    soilwater_flux%Runoff = runoff
    soilwater_flux%LateralFlow = latflow

    ! update soilwater_state
    soilwater_state%WatSto = WatSto
    soilwater_state%PondSto = PondSto
    soilwater_state%Wliq =  spafhy_para%max_poros * min(1.0, (soilwater_state%WatSto / soilwater_state%MaxWatSto))
    soilwater_state%Sat  = soilwater_state%Wliq / spafhy_para%max_poros

    ! HUI - check this here and elsewhere; reduces soil evaporation in drying soil. How done in other models?
    soilwater_state%beta = min(1.0, soilwater_state%Wliq / spafhy_para%max_poros) ! modifier for soil evaporation [-]
    
    call soil_water_retention_curve(soilwater_state%Wliq, spafhy_para, psis) ! soil water potential [MPa]
    soilwater_state%Psi = psis

    call soil_hydraulic_conductivity(soilwater_state%Wliq, spafhy_para, khydr) ! m s-1
    soilwater_state%Kh = khydr

    ! mass balance error [m]
    soilwater_flux%mbe = (soilwater_state%WatSto - WatSto0)  &
                         + (soilwater_state%PondSto - PondSto0) &
                     - (rr - tr - evap - drain - latflow - runoff)
  
  END SUBROUTINE soil_water

  SUBROUTINE soil_water_retention_curve(vol_liq, spafhy_para, smp)
  ! Converts vol. water content to soil water potential (in MPa)
  ! Add restriction that smp can't drop too low?

    real(8), intent(in) :: vol_liq        ! v/v, volumetric of liq in soil bucket
    type(spafhy_para_type), intent(in)    :: spafhy_para ! parameters
    real(8), intent(out):: smp            ! soil suction, negative, MPa

  ! !LOCAL VARIABLES:

    real(8) :: vol_ice     ! v/v, volumetric ice in soil bucket 
    real(8) :: satfrac     ! parameter for Van Genuchten
    real(8) :: n1, m1, alpha_van, watsat, watres 
    real(8) :: eff_porosity! v/v, volume of ice
        
    n1 = spafhy_para%n_van 
    !m1 = 1.0/n1
    m1=1.0-1.0/n1  
    alpha_van = spafhy_para%alpha_van 
    watsat = spafhy_para%watsat  
    watres = spafhy_para%watres
    
    vol_ice = 0.0  
    ! PORT-BRANCH: water.soil_retention.porosity_floor
    ! Condition: watsat - vol_ice < 0.01 -> clamp effective porosity to 0.01
    eff_porosity = max(0.01, watsat - vol_ice)
    
    satfrac = (vol_liq-watres)/(eff_porosity-watres)
    !smp = -(1.0/alpha_van)*(satfrac**(1.0/(m1-1.0)) - 1.0 )**m1 !kPa
    smp = -(1.0/alpha_van)*(satfrac**(1.0/(-m1)) - 1.0 )**(1.0/n1) 
    smp = smp * 0.001 !MPa

  END SUBROUTINE soil_water_retention_curve

  SUBROUTINE soil_hydraulic_conductivity(vol_liq, spafhy_para, khydr)
  ! Computes soil hydraulic conductivity
  ! Equation based on : equation [2] in doi:10.2136/vzj2005.0005 
  !     A Modified Mualem–van Genuchten Formulation for Improved Description of the Hydraulic Conductivity Near Saturation

    real(8), intent(in) :: vol_liq        ! v/v, volumetric of liq in soil bucket
    type(spafhy_para_type), intent(in)    :: spafhy_para ! parameters
    real (8), intent(out) :: khydr      ! hydraulic conductivity [m s-1]

  ! !LOCAL VARIABLES:

    real(8) :: vol_ice     ! v/v, volumetric ice in soil bucket 
    real(8) :: satfrac     ! parameter for Van Genuchten
    real(8) :: n1, m1, alpha_van, watsat,watres ! (-), pore-size-distribution parameter for Van Genuchten 1.07
    real(8) :: eff_porosity! v/v, volume of ice

    n1 = spafhy_para%n_van 
    !m1 = 1.0/n1
    m1 = 1.0-1.0/n1  
    alpha_van = spafhy_para%alpha_van

    watsat = spafhy_para%watsat  
    watres = spafhy_para%watres 
    vol_ice = 0.0  
    ! PORT-BRANCH: water.soil_conductivity.porosity_floor
    ! Condition: watsat - vol_ice < 0.01 -> clamp effective porosity to 0.01
    eff_porosity = max(0.01, watsat - vol_ice)

    satfrac = (vol_liq - watres) / (eff_porosity-watres)

    ! hydraulic conductivity (vanGenuchten - Mualem)
    khydr = spafhy_para%ksat * satfrac**0.5 * (1.0 - (1.0 - satfrac**(1/m1))**m1)**2.0 ! [m s-1]
  
  END SUBROUTINE soil_hydraulic_conductivity

  SUBROUTINE canopy_water_flux(Rn, Ta, Prec, VPD, U, P, fapar, LAI,            & 
                             canopywater_state, canopywater_flux, soilwater_state, spafhy_para)
    !
    ! Computes plant canopy interception, throughfall, ground evaporation, snowpack dynamics
    ! NOTE: Re-think canopy snow interception when snow depth > canopy height 
  
    ! !ARGUMENTS:
    real(8)      , intent(in)    :: Rn     ! Net solar radiation obsorbed by canopy & soil (W m-2)
    real(8)      , intent(in)    :: Ta     ! Air temperature (tc), (degrees C)
    real(8)      , intent(in)    :: Prec   ! precipitatation rate [mm/s] = kg m-2 s-1
    ! real(8)      , intent(in)    :: Par    ! photos. act. radiation [Wm-2]
    real(8)      , intent(in)    :: VPD    ! Vapour pressure deficit (Pa) (will be calculated using pressure & humidity)
    real(8)      , intent(in)    :: U      ! mean wind speed at ref. height above canopy top [ms-1] 
    real(8)      , intent(in)    :: P      ! pressure [Pa], scalar or matrix
    real(8)      , intent(in)    :: fapar  ! fraction of canopy absorbed PAR [-]
    real(8)      , intent(in)    :: LAI    ! leaf area index

    type(canopywater_state_type), intent(inout)   :: canopywater_state  ! canopy state
    type(canopywater_flux_type), intent(inout)    :: canopywater_flux    ! snowpack state
    type(soilwater_state_type), intent(in)        :: soilwater_state  ! soilwater state (for ground evaporation) 
    type(spafhy_para_type), intent(in)    :: spafhy_para

    ! ! Local variables 
    real(8)     ::    Ra, Rb, Ras, ustar, Uh, Ug, fPheno, AE
  
    ! Calculate aerodynamic resistances for canopy and soil layers
    call aerodynamics(LAI, U, Ra, Rb, Ras, ustar, Uh, Ug, spafhy_para)

    ! Calculate canopy interception, canopy evaporation, snowpack dynamics, ground evaporation
    AE = Rn * fapar
    call canopy_water_snow(canopywater_state, canopywater_flux, spafhy_para, Ta, Prec, AE, VPD, Ra, U, LAI, P)

    ! Calculate soil evaporation rate
    AE = Rn * (1 - fapar)
   ! print *, "fapar=", fapar

    call ground_evaporation(canopywater_state, canopywater_flux, soilwater_state, spafhy_para, Ta, AE, VPD, Ras, P)

  END SUBROUTINE canopy_water_flux

  SUBROUTINE ground_evaporation(canopywater_state, canopywater_flux, soilwater_state, spafhy_para, T, AE, VPD, Ras, P)
    !
    ! Calculates evaporation from top soil layer [mm]

    ! !ARGUMENTS:
    real(8)      , intent(in)    :: T      ! air temperature (degC)
    real(8)      , intent(in)    :: AE     ! available energy (~net radiation) (Wm-2)
    real(8)      , intent(in)    :: VPD    ! vapor pressure deficit (Pa)
    real(8)      , intent(in)    :: Ras     ! ground aerodynamic resistance (s m-1)
    real(8)      , intent(in)    :: P      ! pressure [Pa], scalar or matrix
    type(canopywater_flux_type), intent(inout)   :: canopywater_flux
    type(soilwater_state_type), intent(in)          :: soilwater_state  ! soilwater state (for ground evaporation)
    type(canopywater_state_type), intent(in)   :: canopywater_state
    type(spafhy_para_type), intent(in)    :: spafhy_para

    ! ! Local variables 
    real(8)     ::    Lv, erate, Gas, eps=1E-16
    Lv = 1.0e3 * (3147.5 - 2.37 * (T + 273.15))
    Gas = 1 / Ras

    erate = (time_step * 3600) * soilwater_state%beta * penman_monteith(AE, VPD, T, spafhy_para%gsoil, Gas, P) / Lv ! mm

    ! maximum equals available water 
    canopywater_flux%SoilEvap = min(soilwater_state%WatSto, erate)

    ! PORT-BRANCH: water.ground_evaporation.snow_floor_zero
    ! Condition: SWE > eps -> no evaporation from floor if snow on ground
    if (canopywater_state%SWE > eps) then
      canopywater_flux%SoilEvap = 0.0  ! no evaporation from floor if snow on ground
    end if
  END SUBROUTINE ground_evaporation

  SUBROUTINE canopy_water_snow(canopywater_state, canopywater_flux, spafhy_para, T, Pre, AE, D, Ra, U, LAI, P)
    !
    ! Calculates canopy interception, throughfall and snowpack change during timestep dt
    ! Updates canopy and snow storages

    ! !ARGUMENTS:
    real(8)      , intent(in)    :: T      ! air temperature [deg C]
    real(8)      , intent(in)    :: Pre    ! precipitation rate during [mm s-1]
    real(8)      , intent(in)    :: AE     ! available energy (~net radiation) [W m-2]
    real(8)      , intent(in)    :: D      ! vapor pressure deficit [Pa]
    real(8)      , intent(in)    :: Ra     ! canopy aerodynamic resistance [s m-1]
    real(8)      , intent(in)    :: U      ! mean wind speed at ref. height above canopy top [ms-1]
    real(8)      , intent(in)    :: LAI    ! leaf area index [m2 m-2]
    real(8)      , intent(in)    :: P      ! pressure [Pa]
    type(canopywater_state_type), intent(inout)   :: canopywater_state
    type(canopywater_flux_type), intent(inout)    :: canopywater_flux
    type(spafhy_para_type), intent(in)    :: spafhy_para
    
    ! Local variables.
    real(8)  :: fW, fS, Tmin, Tmax, Tmelt, wmax_tot, wmaxsnow_tot
    real(8)  :: Ga, Ce, Sh, gi, erate, gs, Sice=0.0, Sliq=0.0, swe_i, swe_l
    real(8)  :: Lv, Ls, SWEo, Wo, Prec, W
    real(8)  :: eps = 1e-16
    ! local fluxes
    real(8)  :: Unload=0.0, Interc=0.0, Melt=0.0, Freeze=0.0, Trfall=0.0
    real(8)  :: CanopyEvap=0.0, PotInfil=0.0

    ! quality of precipitation depends on temperature
    Tmin = 0.0  ! 'C, below all is snow
    Tmax = 1.0  ! 'C, above all is water
    Tmelt = 0.0  ! 'C, T when melting starts

    ! state of precipitation [as water (fW) or as snow(fS)]
    ! PORT-BRANCH: water.canopy_water_snow.precip_phase
    ! Condition: T<=Tmin -> all snow; T>=Tmax -> all rain; between -> mixed
    if (T <= Tmin) then
      fS = 1.0
      fW = 0.0
    else if (T >= Tmax) then
      fW = 1.0
      fS = 0.0
    else if ((T > Tmin) .and. (T < Tmax)) then
      fW = (T - Tmin) / (Tmax - Tmin)
      fS = 1.0 - fW
    end if

    !canopy storage capacities [mm]
    wmax_tot     = spafhy_para%wmax * LAI
    wmaxsnow_tot = spafhy_para%wmaxsnow * LAI

    ! latent heat of vaporization (Lv) and sublimation (Ls) J kg-1
    Lv = 1.0e3 * (3147.5 - 2.37 * (T + 273.15))
    Ls = Lv + 3.3e5

    Prec = Pre * time_step * 3600  ! mm during timestep


    !----- Initial conditions for calculating mass balance error
    Wo = canopywater_state%CanopyStorage     ! canopy storage at beg. of timestep, mm
    SWEo = canopywater_state%SWE    ! Snow water equivalent mm
    
    !----- Canopy water storage change
    W = Wo
    swe_i = canopywater_state%swe_i
    swe_l = canopywater_state%swe_l

    ! aerodynamic conductance
    Ga = 1.0 / Ra

    ! resistance for snow sublimation adopted from:
    ! Pomeroy et al. 1998 Hydrol proc; Essery et al. 2003 J. Climate;
    ! Best et al. 2011 Geosci. Mod. Dev.

    ! PORT-BRANCH: water.canopy_water_snow.lai_evap_guard
    ! Condition: LAI <= eps -> no canopy evaporation/sublimation
    if ( LAI > eps ) then
      Ce = 0.01*((W + eps) / wmaxsnow_tot)**(-0.4)  ! exposure coeff (-)
      Sh = (1.79 + 3.0*U**0.5)                      ! Sherwood numbner (-)
      gi = Sh*W*Ce / 7.68 + eps                ! m s-1

      erate=0.0
      ! PORT-BRANCH: water.canopy_water_snow.sublim_vs_evap
      ! Condition: Prec==0 & T<=Tmin -> sublimation; Prec==0 & T>Tmin -> evaporation
      if ((Prec == 0) .and. (T <= Tmin)) then
      ! sublimation
        erate =  (time_step * 3600) / Ls * penman_monteith(AE, D, T, gi, Ga, P) ! mm in timestep    
      else if ((Prec == 0) .and. (T > Tmin)) then
      ! evaporation
        gs = 1e6   ! set to large number for free evaporation from wet surface
        erate =  (time_step * 3600) / Lv * penman_monteith(AE, D, T, gs, Ga, P)  ! mm in timestep
      end if 
    else
      erate=0.0
    end if

    ! PORT-BRANCH: water.canopy_water_snow.snow_unloading
    ! Condition: T >= Tmin -> unload excess beyond wmax_tot
    if (T >= Tmin) then
      ! snow unloading from canopy, ensures also that seasonal LAI development does not mess up computations
      Unload = max(W - wmax_tot, 0.0)
      W = W - Unload
    end if

    !----- Interception of rain or snow: asymptotic approach of saturation.
    !      based on: Hedstrom & Pomeroy 1998. Hydrol. Proc 12, 1611-1625;
    !                Koivusalo & Kokkonen 2002 J.Hydrol. 262, 145-164.
    ! PORT-BRANCH: water.canopy_water_snow.interception_phase
    ! Condition: T < Tmin -> snow interception capacity; else -> liquid capacity
    if (T < Tmin) then
      if (LAI > eps) then
        Interc = (wmaxsnow_tot - W) * (1.0 - exp(-Prec/wmaxsnow_tot))
      end if
    elseif (T >= Tmin) then    
    ! Above Tmin, interception capacity equals that of liquid precip
      if (LAI > eps) then
        Interc = max(0.0, (wmax_tot - W)) * (1.0 - exp(-Prec/wmax_tot))
      end if
    end if

    ! update canopy storage after interception
    W = W + Interc ! new canopy storage, mm
    
    ! evaporate from canopy and update storage
    CanopyEvap = min(erate,  W + eps)  ! mm
    W = W - CanopyEvap

    ! Throughfall to field layer or snowpack
    Trfall = Prec + Unload - Interc

    !---- Snowpack (in case no snow, all Trfall routed to floor) """
    ! PORT-BRANCH: water.canopy_water_snow.melt_freeze
    ! Condition: T>=Tmelt -> melt ice; T<Tmelt & swe_l>0 -> freeze liquid; else -> no phase change
    if (T >= Tmelt) then ! .AND. swe_i > 0) then
      Melt = min(swe_i, spafhy_para%kmelt * (time_step*3600) * (T - Tmelt))  ! mm
      Freeze = 0.0
    else if (T < Tmelt .AND. swe_l > 0) then
      Freeze = min(swe_l, spafhy_para%kfreeze * (time_step*3600) * (Tmelt - T))  ! mm
      Melt = 0.0
    else
      Freeze = 0.0
      Melt = 0.0
    end if

    !---- amount of water as ice and liquid in snowpack
    Sice = max(0.0, swe_i + fS * Trfall + Freeze - Melt)
    Sliq = max(0.0, swe_l + fW * Trfall - Freeze + Melt)

    ! The water that can not be hold by snow will penetrate to soil
    PotInfil = max(0.0, Sliq - Sice * spafhy_para%frac_snowliq)  ! mm,
    Sliq = max(0.0, Sliq - PotInfil)  ! mm, liquid water in snow

    ! update canopy state variables
    canopywater_state%CanopyStorage = W
    canopywater_state%swe_l = Sliq
    canopywater_state%swe_i = Sice
    canopywater_state%SWE  = canopywater_state%swe_l + canopywater_state%swe_i

    ! update canopywater_flux
    canopywater_flux%Unloading = Unload   
    canopywater_flux%Interception = Interc
    canopywater_flux%CanopyEvap = CanopyEvap
    canopywater_flux%Throughfall = Trfall
    canopywater_flux%PotInfiltration = PotInfil
    canopywater_flux%Melt = Melt
    canopywater_flux%Freeze = Freeze

    ! mass-balance error mm
    canopywater_flux%mbe = (canopywater_state%CanopyStorage + canopywater_state%SWE) - & 
                               (Wo + SWEo) - (Prec - canopywater_flux%CanopyEvap - & 
                               canopywater_flux%PotInfiltration)

  END SUBROUTINE canopy_water_snow


  SUBROUTINE aerodynamics(LAI, Uo, ra, rb, ras, ustar, Uh, Ug, spafhy_para)
    !
    ! computes wind speed at ground and canopy + boundary layer conductances
    ! Computes wind speed at ground height assuming logarithmic profile above and
    ! exponential within canopy
    ! SOURCE:
    !   Cammalleri et al. 2010 Hydrol. Earth Syst. Sci
    !   Massman 1987, BLM 40, 179 - 197.
    !   Magnani et al. 1998 Plant Cell Env.

    ! !ARGUMENTS:
    real(8)      , intent(in)    :: LAI    ! one-sided leaf-area /plant area index (m2m-2)
    real(8)      , intent(in)    :: Uo     ! mean wind speed at reference height zm (ms-1)
    type(spafhy_para_type), intent(in) :: spafhy_para

    real(8)      , intent(out)   :: ra         ! canopy aerodynamic resistance (sm-1)
    real(8)      , intent(out)   :: rb         ! canopy boundary layer resistance (sm-1)
    real(8)      , intent(out)   :: ras        ! forest floor aerod. resistance (sm-1)
    real(8)      , intent(out)   :: ustar      ! friction velocity (ms-1)
    real(8)      , intent(out)   :: Uh         ! wind speed at hc (ms-1)
    real(8)      , intent(out)   :: Ug         ! wind speed at zg (ms-1)

    ! local variables
    real(8) :: zm1, zg1, alpha1, d, zom, zov, zosv, zn
    real(8) :: kv = 0.4  ! von Karman constant (-)
    real(8) :: beta_aero=285.0   ! s/m, from Campbell & Norman eq. (7.33) x 42.0 molm-3
    real(8) :: eps = 1e-16

    zm1 = spafhy_para%hc + spafhy_para%zmeas  ! m
    ! PORT-BRANCH: water.aerodynamics.ground_height_cap
    ! Condition: zground > 0.1*hc -> cap ground height at 10% of canopy
    zg1 = min(spafhy_para%zground, 0.1 * spafhy_para%hc)
    alpha1 = LAI / 2.0  ! wind attenuation coeff (Yi, 2008 eq. 23)
    d = 0.66*spafhy_para%hc     ! displacement height [m]
    zom = 0.123*spafhy_para%hc  ! roughness lenght for momentum [m]
    zov = 0.1*zom   ! scalar roughness length [m]
    zosv = 0.1*spafhy_para%zo_ground ! soil scalar roughness length [m]

    ! solve ustar and U(hc) from log-profile above canopy
    ustar = Uo * kv / log((zm1 - d) / zom) 
    Uh = ustar / kv * log((spafhy_para%hc - d) / zom)
    
    ! U(zg) from exponential wind profile
    ! PORT-BRANCH: water.aerodynamics.zn_cap
    ! Condition: zg1/hc > 1.0 -> cap normalized ground height at 1.0 (can't exceed canopy top)
    zn = min(zg1 / spafhy_para%hc, 1.0)  ! zground can't be above canopy top
    Ug = Uh * exp(alpha1*(zn - 1.0))

    ! canopy aerodynamic & boundary-layer resistances (sm-1). Magnani et al. 1998 PCE eq. B1 & B5
    !ra = 1. / (kv*ustar) * log((zm - d) / zom)
    ra = 1./(kv**2.0 * Uo) * log((zm1-d)/zom) * log((zm1-d)/zov)   

    ! PORT-BRANCH: water.aerodynamics.rb_lai_guard
    ! Condition: LAI <= eps -> boundary-layer resistance forced to 0 (avoid div-by-zero)
    if (LAI > eps) then 
      rb = 1./LAI * beta_aero * ((spafhy_para%w_leaf / Uh)*(alpha1/(1.0-exp(-alpha1/2.0))))**0.5
    else
      rb = 0.0
    end if

    ! soil aerodynamic resistance (sm-1)
    ras = 1.0/(kv**2.0*Ug) * (log(spafhy_para%zground/spafhy_para%zo_ground))*log(spafhy_para%zground/(zosv))
    
    ra = ra + rb

  END SUBROUTINE aerodynamics


  real(8) FUNCTION penman_monteith(AE, D, T, Gs, Ga, P)

    !------------------------------------------------------
    !  Computes latent heat flux LE (Wm-2) using Penman-Monteith equation
    !  INPUT:
    !     AE - available energy [Wm-2]
    !     D  - vapor pressure deficit [Pa]
    !     T  - ambient air temperature [degC]
    !     Gs - surface (or stomatal) conductance [ms-1]
    !     Ga - aerodynamic conductance [ms-1]
    !     P  - ambient pressure [Pa]
    !  OUTPUT:
    !     x - latent heat flux LE [W m-2]
    !------------------------------------------------------ 
  
    ! !ARGUMENTS
    real(8), intent(in) :: AE                        ! available energy [Wm-2]
    real(8), intent(in) :: D                       ! vapor pressure deficit [Pa]
    real(8), intent(in) :: T                         ! ambient air temperature [degC]
    real(8), intent(in) :: Gs                        ! surface conductance [ms-1]
    real(8), intent(in) :: Ga                        ! aerodynamic conductance [ms-1]
    real(8), intent(in) :: P                         ! ambient pressure [Pa] 
    !character (*), intent(in) :: units          ! W (Wm-2), mm (mms-1=kg m-2 s-1), mol (mol m-2 s-1)

    ! !LOCAL VARIABLES:
    real(8)             :: cp, rho, Mw, s, g, esat, L !P

    ! --- constants
    cp = 1004.67  ! J kg-1 K-1
    rho= 1.25    ! kg m-3
    Mw = 18e-3    ! kg mol-1
    !P = 10130.0  ! standard sea-level pressure (Pa)
  
    call e_sat(T, P, s, g, esat)  ! slope of sat. vapor pressure, psycrom const

    L = 1e3 * (3147.5 - 2.37 * (T + 273.15))

    penman_monteith = (s * AE + rho * cp * Ga * D) / (s + g * (1.0 + Ga / Gs))  ! Wm-2

    ! if (units == 'mm') then
    !   penman_monteith = penman_monteith/L     ! kgm-2s-1 = mms-1
    ! else if (units == 'mol') then
    !   penman_monteith = penman_monteith/L/Mw  ! mol m-2 s-1
    ! end if

    ! PORT-BRANCH: water.penman_monteith.le_floor
    ! Condition: result < 0 -> clamp latent heat flux to 0 (no negative LE)
    penman_monteith = max(penman_monteith, 0.0)

    return

  END FUNCTION penman_monteith

  
  SUBROUTINE e_sat(T, P, s, g, esat)
  !-------------------------------------------
  !  Computes saturation vapor pressure (Pa), slope of vapor pressure curve
  !  [Pa K-1]  and psychrometric constant [Pa K-1]
  !  IN:
  !      T - air temperature (degC)
  !      P - ambient pressure (Pa)
  !  OUT:
  !      esa - saturation vapor pressure in Pa
  !      s - slope of saturation vapor pressure curve (Pa K-1)
  !      g - psychrometric constant (Pa K-1)
  !-------------------------------------------

  ! !ARGUMENTS
    real(8), intent(in) :: T                         ! air temperature (degC)
    real(8), intent(in) :: P                         ! ambient pressure (Pa)
    real(8), intent(out):: esat                       ! saturation vapor pressure in Pa
    real(8), intent(out):: s                         ! slope of saturation vapor pressure curve (Pa K-1)
    real(8), intent(out):: g                         ! psychrometric constant (Pa K-1)
    
  ! !LOCAL VARIABLES:
    real(8)             ::  NT = 273.15
    real(8)             ::  cp = 1004.67             ! J/kg/K
    real(8)             ::  Lambda
    real(8)             ::  eps = 1e-16

    Lambda = 1.0e3 * (3147.5 - 2.37 * (T + NT))         ! lat heat of vapor [J/kg]
    esat = 1.0e3 * (0.6112 * exp((17.67 * T) / (T + 273.16 - 29.66)))  ! Pa

    s = 17.502 * 240.97 * esat / ((240.97 + T) ** 2)
    g = P * cp / (0.622 * Lambda)

  END SUBROUTINE e_sat

end module water_mod
