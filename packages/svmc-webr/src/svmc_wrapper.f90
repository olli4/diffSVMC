! Full SVMC integration loop wrapper for R's .Fortran() interface.
!
! This subroutine replaces the SVMC.f90 main program:
!   - All file I/O (NetCDF, namelists) is replaced by flat array arguments
!   - All numerical computation is identical to the vendor SVMC code
!   - State types are populated from scalar/array R arguments
!   - Output is written to pre-allocated arrays
!
! Provenance: integration loop logic from vendor/SVMC/src/SVMC.f90

subroutine r_svmc_run( &
    ! --- Packed parameter arrays ---
    iparams, rparams, &
    ! --- Hourly climate forcing (length nhours) ---
    temp_hr, rg_hr, prec_hr, vpd_hr, pres_hr, co2_hr, wind_hr, &
    ! --- Daily observations (length ndays) ---
    lai_day_in, snowdepth_day_in, soilmoist_day_in, &
    manage_type_in, manage_c_in_in, manage_c_out_in, &
    ! --- Hourly outputs (length nhours) ---
    gpp_out, gs_out, vcmax_out, jmax_out, &
    chi_out, dpsi_out, le_out, tr_out, &
    soilmoist_out, psi_soil_out, swe_out, &
    ! --- Daily outputs (length ndays) ---
    gpp_day_out, nee_day_out, npp_day_out, &
    hetero_resp_out, auto_resp_out, &
    lai_day_out, cleaf_out, croot_out, cstem_out, &
    soil_c_out, above_bio_out, below_bio_out, &
    status)

  use readctrl_mod
  use readvegpara_mod
  use readsoilpara_mod
  use phydro_mod
  use water_mod
  use wrapper_yasso
  use yasso
  use allocation

  implicit none

  ! --- Packed parameter arrays ---
  ! iparams(8): nhours, ndays, obs_lai, obs_soilmoist, obs_snowdepth,
  !   pft_type_code, invert_option, opt_hypothesis_code
  ! rparams(48): time_step, lat, lon, conductivity, psi50, b, alpha, gamma, rdark,
  !   soil_depth, max_poros, fc, wp, ksat, maxpond, n_van, watres, alpha_van, watsat,
  !   wmax, wmaxsnow, hc, w_leaf, rw, rwmin, gsoil, kmelt, kfreeze, frac_snowliq,
  !   zmeas, zground, zo_ground, cratio_resp, cratio_leaf, cratio_root, cratio_biomass,
  !   harvest_index, turnover_cleaf, turnover_croot, sla, q10,
  !   yasso_totc, yasso_cn_input, yasso_fract_root, yasso_fract_legacy,
  !   yasso_init_temp, yasso_init_temp_ampl, yasso_init_precip
  integer, intent(in)          :: iparams(8)
  double precision, intent(in) :: rparams(48)

  ! --- Unpack dimensions ---
  integer :: nhours, ndays
  integer :: obs_lai_in, obs_soilmoist_in, obs_snowdepth_in
  integer :: pft_type_code, invert_option_in, opt_hypothesis_code
  double precision :: time_step_in, lat_in, lon_in
  double precision :: conductivity_in, psi50_in, b_in, alpha_in, gamma_in, rdark_in
  double precision :: soil_depth_in, max_poros_in, fc_in, wp_in, ksat_in
  double precision :: maxpond_in, n_van_in, watres_in, alpha_van_in, watsat_in
  double precision :: wmax_in, wmaxsnow_in, hc_in, w_leaf_in
  double precision :: rw_in, rwmin_in, gsoil_in
  double precision :: kmelt_in, kfreeze_in, frac_snowliq_in
  double precision :: zmeas_in, zground_in, zo_ground_in
  double precision :: cratio_resp_in, cratio_leaf_in, cratio_root_in, cratio_biomass_in
  double precision :: harvest_index_in, turnover_cleaf_in, turnover_croot_in
  double precision :: sla_in, q10_in
  double precision :: yasso_totc_in, yasso_cn_input_in
  double precision :: yasso_fract_root_in, yasso_fract_legacy_in
  double precision :: yasso_init_temp_in, yasso_init_temp_ampl_in, yasso_init_precip_in

  ! --- Hourly climate forcing ---
  double precision, intent(in) :: temp_hr(*)       ! temperature (K)
  double precision, intent(in) :: rg_hr(*)         ! global radiation (W/m2)
  double precision, intent(in) :: prec_hr(*)       ! precipitation (mm/s)
  double precision, intent(in) :: vpd_hr(*)        ! vapour pressure deficit (Pa)
  double precision, intent(in) :: pres_hr(*)       ! pressure (Pa)
  double precision, intent(in) :: co2_hr(*)        ! CO2 (ppm fraction, multiply by 1e6)
  double precision, intent(in) :: wind_hr(*)       ! wind speed (m/s)

  ! --- Daily observations ---
  double precision, intent(in) :: lai_day_in(*)
  double precision, intent(in) :: snowdepth_day_in(*)
  double precision, intent(in) :: soilmoist_day_in(*)
  integer, intent(in)          :: manage_type_in(*)
  double precision, intent(in) :: manage_c_in_in(*)
  double precision, intent(in) :: manage_c_out_in(*)

  ! --- Hourly outputs ---
  double precision, intent(inout) :: gpp_out(*)
  double precision, intent(inout) :: gs_out(*)
  double precision, intent(inout) :: vcmax_out(*)
  double precision, intent(inout) :: jmax_out(*)
  double precision, intent(inout) :: chi_out(*)
  double precision, intent(inout) :: dpsi_out(*)
  double precision, intent(inout) :: le_out(*)
  double precision, intent(inout) :: tr_out(*)
  double precision, intent(inout) :: soilmoist_out(*)
  double precision, intent(inout) :: psi_soil_out(*)
  double precision, intent(inout) :: swe_out(*)

  ! --- Daily outputs ---
  double precision, intent(inout) :: gpp_day_out(*)
  double precision, intent(inout) :: nee_day_out(*)
  double precision, intent(inout) :: npp_day_out(*)
  double precision, intent(inout) :: hetero_resp_out(*)
  double precision, intent(inout) :: auto_resp_out(*)
  double precision, intent(inout) :: lai_day_out(*)
  double precision, intent(inout) :: cleaf_out(*)
  double precision, intent(inout) :: croot_out(*)
  double precision, intent(inout) :: cstem_out(*)
  double precision, intent(inout) :: soil_c_out(*)
  double precision, intent(inout) :: above_bio_out(*)
  double precision, intent(inout) :: below_bio_out(*)
  integer, intent(out)            :: status

  ! --- Local variables ---
  ! P-hydro outputs
  real(8) :: jmax, dpsi_val, gs, aj, ci, chi_val, vcmax, profit, chi_jmax_lim
  real(8) :: gpp, gpp_day_acc, npp_day_val, nee_day_val
  real(8) :: tr_phydro, tr_spafhy_val
  real(8) :: rn, LE, LatentHeat
  real(8) :: latflow
  real(8) :: psi_soil_val, fapar

  ! State variables
  real(8) :: lai, delta_lai, lai_alloc
  real(8) :: temp_val, rg_val, prec_val, vpd_val, pres_val, co2_val, wind_val
  real(8) :: snowdepth, soilmoist_val
  real(8) :: temp_day_acc, precip_day_acc
  real(8) :: vcmax_day_acc, leaf_rdark_day
  integer :: num_gpp_day, num_vcmax_day

  ! Carbon pools
  real(8) :: croot, cleaf, cstem, cgrain
  real(8) :: above_biomass, below_biomass, yield_val
  real(8) :: leaf_litter_c, root_litter_c, soluble, compost
  real(8) :: HeteroResp, AutoResp_val, TotalResp
  real(8) :: grain_fill_2

  ! Yasso meteorological smoothing
  real(8) :: metyasso(2), metyasso_roll(2)
  integer :: metyasso_ind

  ! State types
  type(soilwater_state_type)   :: soilwater_state
  type(soilwater_flux_type)    :: soilwater_flux
  type(canopywater_state_type) :: canopywater_state
  type(canopywater_flux_type)  :: canopywater_flux
  type(spafhy_para_type)       :: spafhy_para
  type(soilcn_state_type)      :: soilcn_state
  type(soilcn_flux_type)       :: soilcn_flux
  type(yasso_para_type)        :: yasso_para
  type(alloc_para_type)        :: alloc_para
  type(management_data_type)   :: manage_data

  integer :: pheno_stage
  integer :: hr, day_idx, hour_in_day

  status = 0

  ! ===== Unpack parameter arrays =====
  nhours           = iparams(1)
  ndays            = iparams(2)
  obs_lai_in       = iparams(3)
  obs_soilmoist_in = iparams(4)
  obs_snowdepth_in = iparams(5)
  pft_type_code    = iparams(6)
  invert_option_in = iparams(7)
  opt_hypothesis_code = iparams(8)

  time_step_in      = rparams(1)
  lat_in            = rparams(2)
  lon_in            = rparams(3)
  conductivity_in   = rparams(4)
  psi50_in          = rparams(5)
  b_in              = rparams(6)
  alpha_in          = rparams(7)
  gamma_in          = rparams(8)
  rdark_in          = rparams(9)
  soil_depth_in     = rparams(10)
  max_poros_in      = rparams(11)
  fc_in             = rparams(12)
  wp_in             = rparams(13)
  ksat_in           = rparams(14)
  maxpond_in        = rparams(15)
  n_van_in          = rparams(16)
  watres_in         = rparams(17)
  alpha_van_in      = rparams(18)
  watsat_in         = rparams(19)
  wmax_in           = rparams(20)
  wmaxsnow_in       = rparams(21)
  hc_in             = rparams(22)
  w_leaf_in         = rparams(23)
  rw_in             = rparams(24)
  rwmin_in          = rparams(25)
  gsoil_in          = rparams(26)
  kmelt_in          = rparams(27)
  kfreeze_in        = rparams(28)
  frac_snowliq_in   = rparams(29)
  zmeas_in          = rparams(30)
  zground_in        = rparams(31)
  zo_ground_in      = rparams(32)
  cratio_resp_in    = rparams(33)
  cratio_leaf_in    = rparams(34)
  cratio_root_in    = rparams(35)
  cratio_biomass_in = rparams(36)
  harvest_index_in  = rparams(37)
  turnover_cleaf_in = rparams(38)
  turnover_croot_in = rparams(39)
  sla_in            = rparams(40)
  q10_in            = rparams(41)
  yasso_totc_in     = rparams(42)
  yasso_cn_input_in = rparams(43)
  yasso_fract_root_in   = rparams(44)
  yasso_fract_legacy_in = rparams(45)
  yasso_init_temp_in     = rparams(46)
  yasso_init_temp_ampl_in = rparams(47)
  yasso_init_precip_in    = rparams(48)

  ! ===== Set module-level variables from R arguments =====

  ! Control
  time_step = real(time_step_in)
  time_step_output = real(time_step_in)
  obs_lai = (obs_lai_in /= 0)
  obs_soilmoist = (obs_soilmoist_in /= 0)
  obs_snowdepth = (obs_snowdepth_in /= 0)
  obs_manage = .true.
  yasso_year = .false.
  phydro_debug = .false.
  yasso_debug = .false.
  water_debug = .false.
  num_sites = 1
  lat_sites(1) = lat_in
  lon_sites(1) = lon_in

  ! Vegetation parameters
  conductivity = conductivity_in
  psi50 = psi50_in
  b = b_in
  alpha = alpha_in
  gamma = gamma_in
  rdark = rdark_in
  num_pft = 1

  select case (opt_hypothesis_code)
    case (1)
      opt_hypothesis = 'PM'
    case (2)
      opt_hypothesis = 'LC'
    case default
      status = 4
      return
  end select

  select case (pft_type_code)
    case (0)
      pft_type = "other"
    case (1)
      pft_type = "grass"
    case (2)
      pft_type = "oat"
    case default
      pft_type = "other"
  end select

  ! ===== Set spafhy_para from R arguments =====
  spafhy_para%soil_depth    = soil_depth_in
  spafhy_para%max_poros     = max_poros_in
  spafhy_para%fc            = fc_in
  spafhy_para%wp            = wp_in
  spafhy_para%ksat          = ksat_in
  spafhy_para%maxpond       = maxpond_in
  spafhy_para%n_van         = n_van_in
  spafhy_para%watres        = watres_in
  spafhy_para%alpha_van     = alpha_van_in
  spafhy_para%watsat        = watsat_in
  spafhy_para%wmax          = wmax_in
  spafhy_para%wmaxsnow      = wmaxsnow_in
  spafhy_para%hc            = hc_in
  spafhy_para%cf            = 0.6d0
  spafhy_para%w_leaf        = w_leaf_in
  spafhy_para%rw            = rw_in
  spafhy_para%rwmin         = rwmin_in
  spafhy_para%gsoil         = gsoil_in
  spafhy_para%kmelt         = kmelt_in
  spafhy_para%kfreeze       = kfreeze_in
  spafhy_para%frac_snowliq  = frac_snowliq_in
  spafhy_para%zmeas         = zmeas_in
  spafhy_para%zground       = zground_in
  spafhy_para%zo_ground     = zo_ground_in

  ! ===== Set allocation parameters =====
  alloc_para%cratio_resp    = cratio_resp_in
  alloc_para%cratio_leaf    = cratio_leaf_in
  alloc_para%cratio_root    = cratio_root_in
  alloc_para%cratio_biomass = cratio_biomass_in
  alloc_para%harvest_index  = harvest_index_in
  alloc_para%turnover_cleaf = turnover_cleaf_in
  alloc_para%turnover_croot = turnover_croot_in
  alloc_para%sla            = sla_in
  alloc_para%q10            = q10_in
  alloc_para%invert_option  = dble(invert_option_in)

  ! ===== Initialize yasso =====
  call readsoilyasso_namelist(yasso_para)  ! populates from module defaults
  yasso_para%totc = real(yasso_totc_in)
  yasso_para%cn_input = real(yasso_cn_input_in)
  yasso_para%fract_root_input = real(yasso_fract_root_in)
  yasso_para%fract_legacy_soc = real(yasso_fract_legacy_in)
  yasso_para%tempr_c = real(yasso_init_temp_in)
  yasso_para%tempr_ampl = real(yasso_init_temp_ampl_in)
  yasso_para%precip_day = real(yasso_init_precip_in)

  call wrapper_yasso_initialize_totc(soilcn_state, yasso_para, status)
  if (status /= 0) then
    return
  end if

  ! ===== Initialize water state =====
  call initialization_spafhy(canopywater_state, soilwater_state, spafhy_para)

  ! ===== Initialize carbon pools =====
  croot = 0.0d0
  cleaf = 0.0d0
  cstem = 0.0d0
  cgrain = 0.0d0
  above_biomass = 0.0d0
  below_biomass = 0.0d0
  yield_val = 0.0d0
  leaf_litter_c = 0.0d0
  root_litter_c = 0.0d0
  soluble = 0.0d0
  compost = 0.0d0
  pheno_stage = 1
  grain_fill_2 = 0.0d0
  gpp_day_acc = 0.0d0
  vcmax_day_acc = 0.0d0
  temp_day_acc = 0.0d0
  precip_day_acc = 0.0d0
  num_gpp_day = 0
  num_vcmax_day = 0
  metyasso_ind = 1

  ! Initial values
  lai = 0.0d0
  delta_lai = 0.0d0
  lai_alloc = 0.0d0
  fapar = 0.0d0
  psi_soil_val = -1.0d0
  snowdepth = 0.0d0

  ! ===== Main integration loop =====
  day_idx = 0

  do hr = 1, nhours
    hour_in_day = mod(hr - 1, 24)  ! 0..23

    ! --- Read daily input at start of each day ---
    if (hour_in_day == 0) then
      day_idx = day_idx + 1

      if (obs_snowdepth .and. day_idx <= ndays) then
        snowdepth = snowdepth_day_in(day_idx)
      end if

      if (obs_lai .and. day_idx <= ndays) then
        delta_lai = lai_day_in(day_idx) - lai
        lai = lai_day_in(day_idx)
        fapar = 1.0d0 - exp(-k * lai)
      end if

      if (obs_soilmoist .and. day_idx <= ndays) then
        soilmoist_val = soilmoist_day_in(day_idx)
        call soil_water_retention_curve(soilmoist_val, spafhy_para, psi_soil_val)
      end if

      ! Management data
      if (day_idx <= ndays) then
        manage_data%management_type     = manage_type_in(day_idx)
        manage_data%management_c_input  = manage_c_in_in(day_idx)
        manage_data%management_c_output = manage_c_out_in(day_idx)
        manage_data%management_n_input  = 0.0d0
        manage_data%management_n_output = 0.0d0
      end if
    end if

    ! --- Read hourly climate forcing ---
    temp_val = temp_hr(hr)
    rg_val   = rg_hr(hr)
    prec_val = prec_hr(hr)
    vpd_val  = vpd_hr(hr)
    pres_val = pres_hr(hr)
    co2_val  = co2_hr(hr)
    wind_val = wind_hr(hr)

    ! --- Get soil water potential from SpaFHy state if not observed ---
    if (.not. obs_soilmoist) then
      psi_soil_val = soilwater_state%Psi
    end if

    ! --- Run P-hydro (photosynthesis-hydraulics optimization) ---
    if (lai > 0.0d0) then
      call pmodel_hydraulics_numerical( &
          temp_val - 273.15d0, &      ! K -> C
          rg_val * 2.1d0 / lai, &     ! global rad -> PPFD per leaf area
          vpd_val, &
          co2_val * 1.0d6, &          ! fraction -> ppm
          pres_val, &
          fapar, &
          psi_soil_val, &
          rdark, &
          jmax, dpsi_val, gs, aj, ci, chi_val, vcmax, profit, chi_jmax_lim)
    else
      jmax = 0.0d0; dpsi_val = 0.0d0; gs = 0.0d0; aj = 0.0d0
      ci = 0.0d0; chi_val = 0.0d0; vcmax = 0.0d0; profit = 0.0d0
    end if

    ! --- GPP ---
    if (lai > 0.0d0) then
      gpp = (aj + rdark * vcmax) * c_molmass * 1.0d-6 * 1.0d-3 * lai
    else
      gpp = 0.0d0
    end if

    ! --- Transpiration from P-hydro ---
    if (ISNAN(gs) .or. lai <= 0.0d0) then
      tr_phydro = 0.0d0
    else
      tr_phydro = 1.6d0 * gs * (vpd_val / pres_val) * h2o_molmass &
                  / density_h2o(temp_val - 273.15d0, pres_val) * lai
    end if

    ! --- Net radiation ---
    rn = rg_val * 0.7d0

    ! --- SpaFHy canopy water ---
    call reset_spafhy_flux(canopywater_flux, soilwater_flux)
    call canopy_water_flux(rn, temp_val - 273.15d0, prec_val, vpd_val, wind_val, pres_val, &
                           fapar, lai, canopywater_state, canopywater_flux, soilwater_state, &
                           spafhy_para)

    ! --- ET and latent heat ---
    canopywater_flux%ET = tr_phydro * (time_step * 3600.0d0) + canopywater_flux%SoilEvap + &
                          canopywater_flux%CanopyEvap
    LatentHeat = 1.0d3 * (3147.5d0 - 2.37d0 * temp_val)
    LE = canopywater_flux%ET / (time_step * 3600.0d0) * LatentHeat

    ! --- Soil water balance ---
    tr_spafhy_val = tr_phydro * (time_step * 3600.0d0)
    latflow = 0.0d0
    call soil_water(soilwater_state, soilwater_flux, spafhy_para, &
                    canopywater_flux%PotInfiltration, tr_spafhy_val, &
                    canopywater_flux%SoilEvap, latflow)

    ! --- Yasso met smoothing ---
    if (obs_snowdepth .and. snowdepth > 0.01d0) then
      metyasso(1) = real(0.0d0)
    else
      metyasso(1) = real(temp_val - 273.15d0)
    end if
    metyasso(2) = real(prec_val + canopywater_flux%Melt / (time_step * 3600.0d0))

    call exponential_smooth_met(metyasso, metyasso_roll, metyasso_ind, status)
    if (status /= 0) then
      return
    end if
    temp_day_acc = temp_day_acc + dble(metyasso_roll(1))
    precip_day_acc = precip_day_acc + dble(metyasso_roll(2)) * time_step * 3600.0d0

    ! --- Accumulate GPP/Vcmax for daily averages ---
    if (ISNAN(aj) .or. aj < 0.0d0) then
      gpp = 0.0d0
    else
      num_gpp_day = num_gpp_day + 1
    end if
    gpp_day_acc = gpp_day_acc + gpp

    if (ISNAN(vcmax) .or. vcmax <= 0.0d0) then
      vcmax = 0.0d0
    else
      num_vcmax_day = num_vcmax_day + 1
    end if
    vcmax_day_acc = vcmax_day_acc + vcmax

    ! --- Store hourly outputs ---
    gpp_out(hr)       = gpp
    gs_out(hr)        = gs
    vcmax_out(hr)     = vcmax
    jmax_out(hr)      = jmax
    chi_out(hr)       = chi_val
    dpsi_out(hr)      = dpsi_val
    le_out(hr)        = LE
    tr_out(hr)        = tr_phydro
    soilmoist_out(hr) = soilwater_state%Wliq
    psi_soil_out(hr)  = soilwater_state%Psi
    swe_out(hr)       = canopywater_state%SWE

    ! --- Daily sub-calculations at end of day ---
    if (hour_in_day == 23 .and. day_idx <= ndays) then

      temp_day_acc = temp_day_acc / 24.0d0

      if (num_gpp_day == 0) then
        gpp_day_acc = 0.0d0
      else
        gpp_day_acc = gpp_day_acc / num_gpp_day
      end if

      if (num_vcmax_day == 0) then
        vcmax_day_acc = 0.0d0
      else
        vcmax_day_acc = vcmax_day_acc / num_vcmax_day
      end if

      leaf_rdark_day = rdark * vcmax_day_acc * c_molmass * 1.0d-6 * 1.0d-3 * lai

      ! --- Allocation inversion ---
      call invert_alloc(delta_lai, alloc_para, leaf_rdark_day, temp_day_acc, &
                        leaf_litter_c, gpp_day_acc, cleaf, cstem, manage_data, pheno_stage)

      ! --- Carbon allocation ---
      call alloc_hypothesis_2(temp_day_acc, gpp_day_acc, npp_day_val, leaf_rdark_day, &
                              AutoResp_val, croot, cleaf, cstem, cgrain, &
                              leaf_litter_c, root_litter_c, compost, &
                              above_biomass, below_biomass, yield_val, &
                              lai_alloc, alloc_para, grain_fill_2, manage_data, pheno_stage)

      ! --- Yasso soil decomposition ---
      call wrapper_yasso_initialize_flux(soilcn_flux)
      call inputs_to_fractions(leaf_litter_c, root_litter_c, soluble, compost, &
                               soilcn_flux%input_cfract)
      call wrapper_yasso_decompose(soilcn_state, soilcn_flux, yasso_para, &
                                   1.0, real(temp_day_acc), real(precip_day_acc))

      HeteroResp = dble(sum(-soilcn_flux%ctend)) / 24.0d0 / 3600.0d0
      TotalResp = HeteroResp + AutoResp_val / 24.0d0 / 3600.0d0
      nee_day_val = TotalResp - gpp_day_acc

      ! --- Store daily outputs ---
      gpp_day_out(day_idx)     = gpp_day_acc
      nee_day_out(day_idx)     = nee_day_val
      npp_day_out(day_idx)     = npp_day_val
      hetero_resp_out(day_idx) = HeteroResp
      auto_resp_out(day_idx)   = AutoResp_val / 24.0d0 / 3600.0d0
      lai_day_out(day_idx)     = lai_alloc
      cleaf_out(day_idx)       = cleaf
      croot_out(day_idx)       = croot
      cstem_out(day_idx)       = cstem
      soil_c_out(day_idx)      = dble(sum(soilcn_state%cstate))
      above_bio_out(day_idx)   = above_biomass
      below_bio_out(day_idx)   = below_biomass

      ! --- Reset daily accumulators ---
      gpp_day_acc = 0.0d0
      vcmax_day_acc = 0.0d0
      temp_day_acc = 0.0d0
      precip_day_acc = 0.0d0
      num_gpp_day = 0
      num_vcmax_day = 0
    end if
  end do

end subroutine r_svmc_run
