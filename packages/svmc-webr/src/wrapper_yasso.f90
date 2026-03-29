module wrapper_yasso

  use yasso
  use yasso20

  implicit none

  type, public :: soilcn_state_type
    real:: cstate(statesize_yasso)    
    real:: nstate              
  end type soilcn_state_type

  type, public :: soilcn_flux_type
    real:: input_cfract(statesize_yasso)
    real:: input_nfract  
    real:: ctend(statesize_yasso)    
    real:: ntend              
  end type soilcn_flux_type

  type, public :: yasso_para_type
  !----------------------------
  ! Soil properties for yasso
  !-----------------------------
  !   real(8) :: days_yr = 365.0
  !   integer :: statesize_yasso = 5

  ! The Yasso20 maximum a posteriori parameters:
     !integer  :: num_params_y20=35
     real  :: param_y20_map(num_params_y20)
  
  ! Nitrogen-specific parameters
     real(8)  :: nc_mb     ! N-C ratio of the microbial biomass 
     real(8)  :: cue_min   ! minimum microbial carbon use efficiency
     real(8)  :: nc_h_max  ! N-C ratio of the H pool

  ! AWENH composition from Palosuo et al. (2015), for grasses. For now, we'll use the same
  ! composition for both above and below ground inputs. The last values (H) are always 0.
     real(8) :: awenh_fineroot(statesize_yasso)
     real(8) :: awenh_leaf(statesize_yasso)
     real(8) :: awenh_soluble(statesize_yasso)
     real(8) :: awenh_compost(statesize_yasso)

     real :: flux_leafc_day   ! carbon input with "leaf" composition per day
     real :: flux_rootc_day   ! carbon input with "fineroot" composition per day
     real :: flux_nitr_day    ! organic nitrogen input per day
     real :: tempr_c           
     real :: tempr_ampl
     real :: precip_day       ! mm
     real :: totc_min         ! see above       

  ! Get parameter
     real :: alpha_awen(4)
     real :: beta12(2)
     real :: decomp_pc(2)

   ! initialized totc
     real :: totc   ! kg m2
     real :: cn_input
     real :: fract_root_input 
     real :: fract_legacy_soc

  end type yasso_para_type

public readsoilyasso_namelist
public wrapper_yasso_initialize
public wrapper_yasso_initialize_totc
public wrapper_yasso_initialize_flux
public wrapper_yasso_decompose
public exponential_smooth_met

contains

subroutine readsoilyasso_namelist(yasso_para)

  type(yasso_para_type), intent(inout)  :: yasso_para

  ! R-package stub: parameters are set directly by the wrapper.
  ! Populate yasso_para from module-level defaults in yasso module.
  yasso_para%param_y20_map(1:num_params_y20) = param_y20_map
  yasso_para%nc_mb = nc_mb
  yasso_para%cue_min = cue_min
  yasso_para%nc_h_max = nc_h_max
  yasso_para%awenh_fineroot = awenh_fineroot
  yasso_para%awenh_leaf     = awenh_leaf
  yasso_para%awenh_soluble = awenh_soluble
  yasso_para%awenh_compost = awenh_compost

  yasso_para%tempr_c = 5.0
  yasso_para%tempr_ampl = 20.0
  yasso_para%precip_day = 1.8
  yasso_para%totc = 16.0
  yasso_para%cn_input = 50.0
  yasso_para%fract_root_input = 0.5
  yasso_para%fract_legacy_soc = 0.0

  end subroutine readsoilyasso_namelist

  subroutine wrapper_yasso_initialize(soilcn_state, yasso_para)
   
    ! A simple algorithm to initialize the SOC pools into a steady state or a partial
    ! steady-state. First, the equilibrium SOC is evaluated. Then, if totc_min is > 0 and
    ! greater than the equilibrium, the deficit will be covered by increasing the H
    ! pool. The nitrogen pool is left unconstrained and is equal to the equilibrium N +
    ! the possible contribution from the extra H. 

    type(soilcn_state_type), intent(inout)    :: soilcn_state
    type(yasso_para_type), intent(inout)    :: yasso_para

!   call get_params(param_y20_map, yasso_para%alpha_awen, &
 !                     yasso_para%beta12, yasso_para%decomp_pc, yasso_para%param_y20_map)

   call initialize(yasso_para%param_y20_map,  & 
                     yasso_para%flux_leafc_day, &
                     yasso_para%flux_rootc_day, &
                     yasso_para%flux_nitr_day,  &
                     yasso_para%tempr_c,        &
                     yasso_para%precip_day,     &
                     yasso_para%tempr_ampl,     &
                     yasso_para%totc_min,       &
                     soilcn_state%cstate,       &
                     soilcn_state%nstate) 

  end subroutine wrapper_yasso_initialize

  subroutine wrapper_yasso_initialize_totc(soilcn_state, yasso_para)
    ! Another, simpler initialization method which enforces the total C and N stocks
    ! strictly and requires setting the fraction of "legacy" carbon explicitly. Given a
    ! total C, the C pools are set as a weighted combination of an equilibrated
    ! partitioning and a "legacy" partitioning where all C is assigned to the H pool. The
    ! weighting is given by the fract_legacy_soc parameter. The N pool is set analoguously
    ! with the equilibrium N depending on the given C:N ratio of input.
    type(soilcn_state_type), intent(inout)    :: soilcn_state
    type(yasso_para_type), intent(inout)    :: yasso_para

   !call get_params(param_y20_map, yasso_para%alpha_awen, &
   !                   yasso_para%beta12, yasso_para%decomp_pc, yasso_para%param_y20_map)

   call initialize_totc(yasso_para%param_y20_map,  & 
                        yasso_para%totc, & 
                        yasso_para%cn_input,  & 
                        yasso_para%fract_root_input,  & 
                        yasso_para%fract_legacy_soc,  &
                        yasso_para%tempr_c,        &
                        yasso_para%precip_day,     &
                        yasso_para%tempr_ampl,     &
                        soilcn_state%cstate,       &
                        soilcn_state%nstate) 

   end subroutine wrapper_yasso_initialize_totc

   subroutine wrapper_yasso_initialize_flux(soilcn_flux)
     type(soilcn_flux_type), intent(inout)    :: soilcn_flux

     soilcn_flux%input_cfract(:)=0.0
     soilcn_flux%input_nfract=0.0
     soilcn_flux%ctend(:)=0.0
     soilcn_flux%ntend=0.0
   end subroutine wrapper_yasso_initialize_flux

  subroutine wrapper_yasso_decompose(soilcn_state, soilcn_flux, yasso_para, & 
                       timestep_days, tempr_c, precip_day)
    type(soilcn_state_type), intent(inout)    :: soilcn_state
    type(soilcn_flux_type), intent(inout)    :: soilcn_flux
    type(yasso_para_type), intent(in)    :: yasso_para
    real, intent(in) :: timestep_days
    real, intent(in) :: tempr_c ! air temperature
    real, intent(in) :: precip_day ! precipitation mm / day
    
    call decompose(yasso_para%param_y20_map, timestep_days, tempr_c, &
                   precip_day, soilcn_state%cstate, soilcn_state%nstate, & 
                   soilcn_flux%ctend, soilcn_flux%ntend)
    
    soilcn_state%cstate=soilcn_state%cstate + soilcn_flux%ctend + soilcn_flux%input_cfract
    soilcn_state%nstate=soilcn_state%nstate + soilcn_flux%ntend + soilcn_flux%input_nfract

  end subroutine wrapper_yasso_decompose

  subroutine wrapper_yasso_annual(soilcn_state, soilcn_flux, yasso_para, & 
                       timestep_days, days_yr, temp_mon, precip_year)
    type(soilcn_state_type), intent(inout)    :: soilcn_state
    type(soilcn_flux_type), intent(inout)    :: soilcn_flux
    type(yasso_para_type), intent(in)    :: yasso_para
    real, intent(in) :: timestep_days
    real, intent(in) :: days_yr
    real, dimension(12), intent(in) :: temp_mon ! air temperature
    real, intent(in) :: precip_year ! precipitation mm / day

    ! local variables
    real ::  leac, d
    real ::  timestep_yr
    real,dimension(5) :: cstate_next ! keep soil carbon state for the next time step
    logical :: steadystate_pred ! switch to turn on steady-state assumption
    
    timestep_yr = timestep_days / days_yr
    leac=0.0  ! leaching unit?
    d=0.0     ! size effect of wood debris on decomposition
              ! for grass/crop this can be zero.
    steadystate_pred=.False.  

    call mod5c20(yasso_para%param_y20_map, timestep_yr, temp_mon, &
                   precip_year, soilcn_state%cstate, soilcn_flux%input_cfract, d, &
                   leac, cstate_next, steadystate_pred)
    
    soilcn_flux%ctend=cstate_next-(soilcn_state%cstate+soilcn_flux%input_cfract)
    soilcn_state%cstate=cstate_next

  end subroutine wrapper_yasso_annual


  subroutine exponential_smooth_met(met_daily, met_rolling, met_ind)
    ! Evaluate an expotential smoothing. used for scaling met
    ! parameters from daily to monthly level.
    ! https://en.wikipedia.org/wiki/Exponential_smoothing
    real, intent(in) :: met_daily(:)         ! current meteorological data
    real, intent(inout) :: met_rolling(:)    ! previous-step meteorological data
    integer, intent(inout) :: met_ind     ! a counter, must be 1 on first call, not changed outside
    ! local variables
    real           :: alpha_smooth1=0.01, alpha_smooth2=0.0016

    ! PORT-BRANCH: yasso.exponential_smooth_met.invalid_ind_guard
    ! Condition: met_ind < 1 -> fatal error (invalid counter state)
    if (met_ind < 1 ) then
       ! print suppressed for WASM compat
       return
    end if
    
    ! PORT-BRANCH: yasso.exponential_smooth_met.init_vs_smooth
    ! Condition: met_ind == 1 -> initialize rolling from daily; else -> exponential smoothing
    if (met_ind == 1) then
       ! For the first aver_size days average as many values as have been input.
       met_rolling(:) = met_daily(:)
       met_ind = met_ind + 1
    else
       ! met_ind now stays as aver_size+1
       met_rolling(1) = alpha_smooth1 * met_daily(1) + (1-alpha_smooth1) * met_rolling(1)
       met_rolling(2) = alpha_smooth2 * met_daily(2) + (1-alpha_smooth2) * met_rolling(2)
    end if

  end subroutine exponential_smooth_met

  
end module wrapper_yasso

