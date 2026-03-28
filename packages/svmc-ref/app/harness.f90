PROGRAM harness
  !-----------------------------------------------------------------
  ! Test harness for SVMC leaf-level functions.
  ! Calls each function over a grid of inputs and writes JSONL
  ! (one JSON object per line) to fixtures.jsonl via unit 10.
  !
  ! stdout is reserved for the human-readable record count summary.
  ! This prevents Fortran runtime warnings from contaminating the
  ! machine-readable JSONL stream.
  !
  ! Private functions from water_mod (e_sat, penman_monteith) are
  ! duplicated here since they cannot be accessed via USE.
  !-----------------------------------------------------------------

  use phydro_mod, only: ftemp_arrh, gammastar, ftemp_kphio, density_h2o, &
                       viscosity_h2o, calc_kmm, scale_conductivity, &
                       calc_gs, calc_assim_light_limited, fn_profit, &
                       quadratic, pmodel_hydraulics_numerical
  use water_mod, only: soil_water_retention_curve, soil_hydraulic_conductivity, &
                       canopy_water_flux, soil_water
  use yasso, only: inputs_to_fractions, statesize_yasso, decompose, &
                    nc_mb, cue_min, nc_h_max, param_y20_map, &
                    initialize_totc, awenh_fineroot, awenh_leaf
  use yasso20, only: matrixexp, matrixnorm, mod5c20
  use readvegpara_mod, only: par_plant_type, par_cost_type, par_env_type, &
                             par_photosynth_type, optimizer_type, kphio, &
                             opt_hypothesis, pft_type, conductivity, psi50, &
                             b, alpha, gamma
  use readsoilpara_mod, only: spafhy_para_type, canopywater_state_type, &
                              canopywater_flux_type, soilwater_state_type, &
                              soilwater_flux_type
  use readctrl_mod, only: time_step
  use allocation, only: alloc_para_type, management_data_type, &
                        alloc_hypothesis_2, invert_alloc

  implicit none

  ! --- Test-grid dimensions ---
  integer, parameter :: NTC = 10
  integer, parameter :: NP  = 3
  integer, parameter :: NVPD = 4
  integer, parameter :: NDHA = 3
  integer, parameter :: NTHETA = 5

  real(8) :: tc_grid(NTC)
  real(8) :: patm_grid(NP)
  real(8) :: vpd_grid(NVPD)
  real(8) :: dha_grid(NDHA)
  real(8) :: theta_grid(NTHETA)

  ! Working variables
  real(8) :: tc, patm, vpd_val, dha, theta
  real(8) :: tk, result_d
  real(8) :: kmm_out
  real(8) :: esat_out, s_out, g_out
  real(8) :: smp_out, khydr_out
  real(8) :: le_out
  real(8) :: ci_out, aj_out

  ! quadratic test cases
  real(8) :: quad_r1
  integer, parameter :: NQUAD = 9
  real(8) :: quad_a(NQUAD), quad_b(NQUAD), quad_c(NQUAD)

  ! inputs_to_fractions
  real(8) :: fract_out(statesize_yasso)
  integer, parameter :: NFRAC = 5

  ! exponential_smooth_met
  real(8) :: met_daily(2), met_rolling(2)
  integer :: met_ind

  ! aerodynamics
  integer, parameter :: NLAI_AERO = 4, NUO_AERO = 3, NPARA_AERO = 2
  real(8) :: lai_aero_grid(NLAI_AERO), uo_aero_grid(NUO_AERO)
  real(8) :: aero_ra, aero_rb, aero_ras, aero_ustar, aero_Uh, aero_Ug
  type(spafhy_para_type) :: aero_para(NPARA_AERO)

  type(par_plant_type)      :: par_plant
  type(par_cost_type)       :: par_cost
  type(par_env_type)        :: par_env
  type(par_photosynth_type) :: par_photosynth
  type(optimizer_type)      :: lj_dps
  type(spafhy_para_type)    :: soil_params
  type(spafhy_para_type)    :: soil_floor_params
  type(spafhy_para_type)    :: aero_cap_params

  ! Phase 3 canopy/soil state and flux variables
  type(canopywater_state_type)  :: cw_state
  type(canopywater_flux_type)   :: cw_flux
  type(soilwater_state_type)    :: sw_state
  type(soilwater_flux_type)     :: sw_flux
  type(spafhy_para_type)        :: cwf_para  ! canopy water flux parameters

  ! Phase 3 test-case dimensions
  integer, parameter :: NCWF = 8   ! canopy_water_flux test cases
  real(8) :: cwf_Rn(NCWF), cwf_Ta(NCWF), cwf_Prec(NCWF)
  real(8) :: cwf_VPD(NCWF), cwf_U(NCWF), cwf_P(NCWF)
  real(8) :: cwf_fapar(NCWF), cwf_LAI(NCWF)
  ! Initial canopy state for each case
  real(8) :: cwf_CanSto0(NCWF), cwf_SWE0(NCWF)
  real(8) :: cwf_swe_i0(NCWF), cwf_swe_l0(NCWF)
  ! Initial soil state for each case
  real(8) :: cwf_WatSto0(NCWF), cwf_beta0(NCWF)

  ! soil_water test-case dimensions
  integer, parameter :: NSW = 5    ! soil_water test cases
  real(8) :: sw_potinf(NSW), sw_tr(NSW), sw_evap(NSW), sw_latflow(NSW)
  real(8) :: sw_WatSto0(NSW), sw_PondSto0(NSW)
  real(8) :: sw_Wliq0(NSW), sw_Kh0(NSW), sw_FcSto0(NSW)
  real(8) :: sw_MaxWatSto0(NSW), sw_MaxPondSto0(NSW)
  real(8) :: sw_tr_out, sw_evap_out, sw_latflow_out

  ! pmodel_hydraulics_numerical outputs
  real(8) :: phn_jmax, phn_dpsi, phn_gs, phn_aj, phn_ci, phn_chi
  real(8) :: phn_vcmax, phn_profit, phn_chi_jmax_lim
  integer, parameter :: NPHN = 6
  real(8) :: phn_tc(NPHN), phn_vpd(NPHN), phn_ppfd(NPHN)
  real(8) :: phn_co2(NPHN), phn_psi_soil(NPHN), phn_fapar(NPHN)
  real(8) :: phn_sp(NPHN), phn_rdark(NPHN)

  ! matrixexp / matrixnorm test matrices (5×5)
  integer, parameter :: NMEXP = 8
  real(8) :: mexp_A(5,5), mexp_B(5,5)
  real(8) :: mnorm_p
  real(8) :: mexp_theta(35), mexp_temp(12)
  real(8) :: mexp_time, mexp_prec, mexp_d, mexp_leac

  ! mod5c20 test variables
  real(8) :: m5c_init(5), m5c_b(5), m5c_xt(5)

  ! decompose test variables
  integer, parameter :: NDECOMP = 10
  real(8) :: dec_param(35)
  real(8) :: dec_timestep_days
  real(8) :: dec_tempr_c(NDECOMP)
  real(8) :: dec_precip_day(NDECOMP)
  real(8) :: dec_cstate(5,NDECOMP)
  real(8) :: dec_nstate(NDECOMP)
  real(8) :: dec_ctend(5), dec_ntend

  ! initialize_totc test variables
  integer, parameter :: NINIT_TOTC = 8
  real(8) :: itc_totc(NINIT_TOTC)
  real(8) :: itc_cn_input(NINIT_TOTC)
  real(8) :: itc_fract_root(NINIT_TOTC)
  real(8) :: itc_fract_legacy(NINIT_TOTC)
  real(8) :: itc_tempr_c(NINIT_TOTC)
  real(8) :: itc_precip_day(NINIT_TOTC)
  real(8) :: itc_tempr_ampl(NINIT_TOTC)
  real(8) :: itc_cstate_out(5), itc_nstate_out

  ! --- Allocation test variables (Phase 4b) ---
  integer, parameter :: N_AH2 = 13        ! alloc_hypothesis_2 test cases
  integer, parameter :: N_INVALLOC = 14   ! invert_alloc test cases
  type(alloc_para_type) :: ah2_ap
  type(management_data_type) :: ah2_md
  real(8) :: ah2_temp, ah2_gpp, ah2_rdark
  real(8) :: ah2_npp, ah2_resp, ah2_croot, ah2_cleaf, ah2_cstem, ah2_cgrain
  real(8) :: ah2_litter_cleaf, ah2_litter_croot, ah2_compost
  real(8) :: ah2_lai, ah2_above, ah2_below, ah2_yield, ah2_grain_fill
  integer :: ah2_pheno
  real(8) :: ah2_sin(14), ah2_sout(14)  ! state snapshot arrays
  integer :: ah2_pheno_in                ! saved input pheno_stage
  character(len=256) :: ah2_pft_saved
  ! invert_alloc
  type(alloc_para_type) :: inv_ap
  type(management_data_type) :: inv_md
  real(8) :: inv_delta_lai, inv_rdark, inv_temp, inv_gpp
  real(8) :: inv_litter_cleaf, inv_cleaf, inv_cstem
  integer :: inv_pheno
  real(8) :: inv_sin(5), inv_sout(8)  ! state/alloc output snapshots

  integer :: i, j, k, m
  integer :: nrec  ! record counter

  ! --- Open JSONL output file on unit 10 ---
  open(unit=10, file='fixtures.jsonl', status='replace', action='write')
  nrec = 0

  ! --- Initialise grids ---
  tc_grid   = (/ -10.0d0, 0.0d0, 5.0d0, 10.0d0, 15.0d0, &
                  20.0d0, 25.0d0, 30.0d0, 35.0d0, 40.0d0 /)
  patm_grid = (/ 80000.0d0, 101325.0d0, 110000.0d0 /)
  vpd_grid  = (/ 500.0d0, 1000.0d0, 2000.0d0, 3000.0d0 /)
  dha_grid  = (/ 37830.0d0, 50000.0d0, 79430.0d0 /)
  theta_grid = (/ 0.10d0, 0.20d0, 0.30d0, 0.40d0, 0.44d0 /)

  ! quadratic(a,b,c) test inputs — covers standard, a=0, discriminant~0, b=0,
  ! and near-zero negative discriminant (case 9: b²-4ac barely < 0 → tolerance clamp)
  quad_a = (/ -1.0d0, -2.0d0,  1.0d0, -0.5d0,  0.0d0, 0.0d0, -1.0d0, -3.0d0, 1.0d0 /)
  quad_b = (/  5.0d0,  3.0d0, -4.0d0,  0.1d0,  2.0d0, 0.0d0,  2.0d0,  6.0d0, 1.0d0 /)
  quad_c = (/ -6.0d0,  1.0d0,  4.0d0,  0.0d0, -3.0d0, 0.0d0,  1.0d0, -3.0d0, 0.0d0 /)
  quad_c(9) = nearest(0.25d0, 1.0d0)  ! next representable float above 0.25

  ! --- Set module-level variables needed by phydro_mod ---
  ! (these are normally set by readvegpara_namelist)
  ! We don't call the namelist readers — just set values directly.
  time_step = 1.0       ! readctrl_mod: hours
  opt_hypothesis = 'PM' ! readvegpara_mod: profit maximisation

  ! --- Typical soil parameters (Launiainen et al. 2022 class C1) ---
  soil_params%n_van     = 1.12d0
  soil_params%alpha_van = 4.45d0
  soil_params%watsat    = 0.75d0
  soil_params%watres    = 0.0d0
  soil_params%ksat      = 1.0d-5  ! m/s
  soil_params%soil_depth = 1.0d0
  soil_params%max_poros  = 0.75d0

  soil_floor_params = soil_params
  soil_floor_params%watsat = 0.005d0
  soil_floor_params%max_poros = 0.005d0

  ! --- Aerodynamics test grids ---
  lai_aero_grid = (/ 0.0d0, 0.5d0, 2.0d0, 6.0d0 /)
  uo_aero_grid  = (/ 0.5d0, 2.0d0, 5.0d0 /)
  ! Parameter set 1: conifer
  aero_para(1)%hc       = 15.0d0
  aero_para(1)%zmeas    = 2.0d0
  aero_para(1)%zground  = 0.5d0
  aero_para(1)%zo_ground = 0.01d0
  aero_para(1)%w_leaf   = 0.02d0
  ! Parameter set 2: broadleaf
  aero_para(2)%hc       = 20.0d0
  aero_para(2)%zmeas    = 5.0d0
  aero_para(2)%zground  = 1.0d0
  aero_para(2)%zo_ground = 0.05d0
  aero_para(2)%w_leaf   = 0.05d0

  aero_cap_params = aero_para(1)
  aero_cap_params%hc = 10.0d0
  aero_cap_params%zmeas = 2.0d0
  aero_cap_params%zground = 2.0d0
  aero_cap_params%zo_ground = 0.02d0
  aero_cap_params%w_leaf = 0.02d0

  ! --- Phase 3: SpaFHy parameters for canopy/soil water tests ---
  ! Use conifer parameter set as base
  cwf_para = aero_para(1)
  cwf_para%wmax        = 0.5d0     ! canopy liquid water storage capacity per LAI, mm
  cwf_para%wmaxsnow    = 4.0d0     ! canopy snow storage capacity per LAI, mm
  cwf_para%gsoil       = 0.01d0    ! soil surface conductance, m/s
  cwf_para%kmelt       = 2.31d-8   ! snowmelt rate, mm / (s K)
  cwf_para%kfreeze     = 5.79d-9   ! freeze rate, mm / (s K)
  cwf_para%frac_snowliq = 0.05d0   ! max fraction of liquid water in snow
  cwf_para%rw          = 0.0d0
  cwf_para%rwmin       = 0.0d0
  cwf_para%cf          = 0.6d0
  ! Soil params from existing soil_params
  cwf_para%n_van       = soil_params%n_van
  cwf_para%alpha_van   = soil_params%alpha_van
  cwf_para%watsat      = soil_params%watsat
  cwf_para%watres      = soil_params%watres
  cwf_para%ksat        = soil_params%ksat
  cwf_para%soil_depth  = soil_params%soil_depth
  cwf_para%max_poros   = soil_params%max_poros
  cwf_para%maxpond     = 10.0d0    ! max pond storage, mm

  ! canopy_water_flux test cases (8 scenarios):
  !   1: warm rain, dry canopy          2: cold snow, existing canopy storage
  !   3: no precip, wet canopy (evap)   4: freezing, existing snowpack
  !   5: mild, LAI=0 (bare ground)      6: high precip, saturated canopy
  !   7: mixed precip, warm snowmelt    8: cold snowpack, no phase change
  !       Rn(W/m2)  Ta(C)    Prec(mm/s) VPD(Pa) U(m/s) P(Pa)     fapar  LAI
  cwf_Rn    = (/ 200.0d0,  50.0d0,  150.0d0,  10.0d0, 200.0d0, 300.0d0, 120.0d0,  35.0d0 /)
  cwf_Ta    = (/  15.0d0,  -5.0d0,   20.0d0,  -2.0d0,  10.0d0,  18.0d0,   0.5d0,  -1.5d0 /)
  cwf_Prec  = (/ 1.4d-4,  2.8d-4,   0.0d0,   0.0d0, 5.6d-4, 8.3d-4, 1.5d-4,   0.0d0 /)
  cwf_VPD   = (/ 800.0d0, 200.0d0, 1200.0d0, 300.0d0, 600.0d0, 500.0d0, 450.0d0, 250.0d0 /)
  cwf_U     = (/   2.0d0,   3.0d0,    1.5d0,   4.0d0,   2.5d0,   2.0d0,   2.2d0,   1.8d0 /)
  cwf_P     = (/ 101325.0d0, 101325.0d0, 101325.0d0, 80000.0d0, 101325.0d0, 101325.0d0, 101325.0d0, 95000.0d0 /)
  cwf_fapar = (/   0.8d0,   0.7d0,    0.9d0,   0.6d0,   0.0d0,   0.85d0,   0.75d0,   0.65d0 /)
  cwf_LAI   = (/   3.0d0,   2.0d0,    4.0d0,   1.5d0,   0.0d0,   5.0d0,    2.5d0,   1.2d0 /)
  ! Initial canopy state
  cwf_CanSto0 = (/  0.0d0,  0.5d0,   1.2d0,   0.0d0,   0.0d0,   2.0d0,   0.1d0,   0.3d0 /)
  cwf_SWE0    = (/  0.0d0,  5.0d0,   0.0d0,  10.0d0,   0.0d0,   0.0d0,   3.0d0,   3.0d0 /)
  cwf_swe_i0  = (/  0.0d0,  4.5d0,   0.0d0,   9.0d0,   0.0d0,   0.0d0,   2.8d0,   3.0d0 /)
  cwf_swe_l0  = (/  0.0d0,  0.5d0,   0.0d0,   1.0d0,   0.0d0,   0.0d0,   0.2d0,   0.0d0 /)
  ! Initial soil state
  cwf_WatSto0 = (/ 100.0d0, 80.0d0, 150.0d0, 200.0d0, 120.0d0, 100.0d0, 110.0d0,  90.0d0 /)
  cwf_beta0   = (/   0.8d0,  0.6d0,   1.0d0,   0.9d0,   0.7d0,   0.8d0,   0.75d0,  0.65d0 /)

  ! soil_water test cases (5 scenarios):
  !   1: normal infiltration   2: excess water → ponding
  !   3: dry soil, low precip  4: saturated, heavy rain → runoff
  !   5: transpiration-dominated
  sw_potinf     = (/  5.0d0,  20.0d0,   1.0d0,  50.0d0,   2.0d0  /)
  sw_tr         = (/  0.5d0,   0.2d0,   0.1d0,   0.0d0,   3.0d0  /)
  sw_evap       = (/  0.3d0,   0.1d0,   0.05d0,  0.0d0,   0.5d0  /)
  sw_latflow    = (/  0.0d0,   0.0d0,   0.0d0,   0.0d0,   0.0d0  /)
  sw_WatSto0    = (/ 300.0d0, 700.0d0,  50.0d0, 750.0d0, 400.0d0  /)
  sw_PondSto0   = (/   0.0d0,   5.0d0,   0.0d0,   0.0d0,   0.0d0  /)
  sw_Wliq0      = (/   0.3d0,  0.65d0,   0.05d0,  0.75d0,  0.4d0  /)
  sw_Kh0        = (/ 1.0d-6,  5.0d-6,   1.0d-8,  1.0d-5,  2.0d-6  /)
  sw_FcSto0     = (/ 200.0d0, 200.0d0, 200.0d0, 200.0d0, 200.0d0  /)
  sw_MaxWatSto0 = (/ 750.0d0, 750.0d0, 750.0d0, 750.0d0, 750.0d0  /)
  sw_MaxPondSto0= (/  10.0d0,  10.0d0,  10.0d0,  10.0d0,  10.0d0  /)

  ! ================================================================
  ! 1. ftemp_arrh(tk, dha)
  ! ================================================================
  do i = 1, NTC
    tc = tc_grid(i)
    tk = tc + 273.15d0
    do k = 1, NDHA
      dha = dha_grid(k)
      result_d = ftemp_arrh(tk, dha)
      write(10, '(A,F10.4,A,F10.1,A,ES22.15,A)') &
        '{"fn":"ftemp_arrh","inputs":{"tk":', tk, ',"dha":', dha, &
        '},"output":', result_d, '}'
      nrec = nrec + 1
    end do
  end do

  ! ================================================================
  ! 2. gammastar(tc, patm)
  ! ================================================================
  do i = 1, NTC
    tc = tc_grid(i)
    do j = 1, NP
      patm = patm_grid(j)
      result_d = gammastar(tc, patm)
      write(10, '(A,F10.4,A,F12.1,A,ES22.15,A)') &
        '{"fn":"gammastar","inputs":{"tc":', tc, ',"patm":', patm, &
        '},"output":', result_d, '}'
      nrec = nrec + 1
    end do
  end do

  ! ================================================================
  ! 3. ftemp_kphio(tc, c4) — C3 and C4
  ! ================================================================
  do i = 1, NTC
    tc = tc_grid(i)
    result_d = ftemp_kphio(tc, .FALSE.)
    write(10, '(A,F10.4,A,ES22.15,A)') &
      '{"fn":"ftemp_kphio_c3","inputs":{"tc":', tc, &
      '},"output":', result_d, '}'
    nrec = nrec + 1
    result_d = ftemp_kphio(tc, .TRUE.)
    write(10, '(A,F10.4,A,ES22.15,A)') &
      '{"fn":"ftemp_kphio_c4","inputs":{"tc":', tc, &
      '},"output":', result_d, '}'
    nrec = nrec + 1
  end do

  ! ================================================================
  ! 4. density_h2o(tc, patm)
  ! ================================================================
  do i = 1, NTC
    tc = tc_grid(i)
    do j = 1, NP
      patm = patm_grid(j)
      result_d = density_h2o(tc, patm)
      write(10, '(A,F10.4,A,F12.1,A,ES22.15,A)') &
        '{"fn":"density_h2o","inputs":{"tc":', tc, ',"patm":', patm, &
        '},"output":', result_d, '}'
      nrec = nrec + 1
    end do
  end do

  ! ================================================================
  ! 5. viscosity_h2o(tc, patm)
  ! ================================================================
  do i = 1, NTC
    tc = tc_grid(i)
    do j = 1, NP
      patm = patm_grid(j)
      result_d = viscosity_h2o(tc, patm)
      write(10, '(A,F10.4,A,F12.1,A,ES22.15,A)') &
        '{"fn":"viscosity_h2o","inputs":{"tc":', tc, ',"patm":', patm, &
        '},"output":', result_d, '}'
      nrec = nrec + 1
    end do
  end do

  ! ================================================================
  ! 6. calc_kmm(tc, patm) — subroutine with intent(out) kmm
  ! ================================================================
  do i = 1, NTC
    tc = tc_grid(i)
    do j = 1, NP
      patm = patm_grid(j)
      call calc_kmm(tc, patm, kmm_out)
      write(10, '(A,F10.4,A,F12.1,A,ES22.15,A)') &
        '{"fn":"calc_kmm","inputs":{"tc":', tc, ',"patm":', patm, &
        '},"output":', kmm_out, '}'
      nrec = nrec + 1
    end do
  end do

  ! ================================================================
  ! 7. e_sat(T, P) → (s, g, esat) — PRIVATE in water_mod, duplicated here
  ! ================================================================
  do i = 1, NTC
    tc = tc_grid(i)
    do j = 1, NP
      patm = patm_grid(j)
      call harness_e_sat(tc, patm, s_out, g_out, esat_out)
      write(10, '(A,F10.4,A,F12.1,A,ES22.15,A,ES22.15,A,ES22.15,A)') &
        '{"fn":"e_sat","inputs":{"tc":', tc, ',"patm":', patm, &
        '},"output":{"esat":', esat_out, ',"s":', s_out, ',"g":', g_out, '}}'
      nrec = nrec + 1
    end do
  end do

  ! ================================================================
  ! 8. penman_monteith(AE, D, T, Gs, Ga, P) — PRIVATE, duplicated here
  ! ================================================================
  do i = 1, NTC
    tc = tc_grid(i)
    do j = 1, NP
      patm = patm_grid(j)
      do m = 1, NVPD
        vpd_val = vpd_grid(m)
        le_out = harness_penman_monteith(200.0d0, vpd_val, tc, 0.01d0, 0.05d0, patm)
        write(10, '(A,F10.4,A,F12.1,A,F10.1,A,ES22.15,A)') &
          '{"fn":"penman_monteith","inputs":{"AE":200.0,"tc":', tc, &
          ',"patm":', patm, ',"vpd":', vpd_val, &
          ',"Gs":0.01,"Ga":0.05},"output":', le_out, '}'
        nrec = nrec + 1
      end do
    end do
  end do

  le_out = harness_penman_monteith(-50.0d0, 0.0d0, 20.0d0, 0.01d0, 0.05d0, 101325.0d0)
  write(10, '(A,ES22.15,A)') &
    '{"fn":"penman_monteith","inputs":{"AE":-50.0,"tc":20.0,' // &
    '"patm":101325.0,"vpd":0.0,"Gs":0.01,"Ga":0.05},"output":', le_out, '}'
  nrec = nrec + 1

  ! ================================================================
  ! 9. soil_water_retention_curve(vol_liq, spafhy_para, smp)
  ! ================================================================
  do k = 1, NTHETA
    theta = theta_grid(k)
    call soil_water_retention_curve(theta, soil_params, smp_out)
    write(10, '(A,F8.4,A,ES22.15,A)') &
      '{"fn":"soil_water_retention_curve","inputs":{"vol_liq":', theta, &
      ',"n_van":1.12,"alpha_van":4.45,"watsat":0.75,"watres":0.0},"output":', smp_out, '}'
    nrec = nrec + 1
  end do

  call soil_water_retention_curve(0.004d0, soil_floor_params, smp_out)
  write(10, '(A,ES22.15,A)') &
    '{"fn":"soil_water_retention_curve","inputs":{"vol_liq":0.004,' // &
    '"n_van":1.12,"alpha_van":4.45,"watsat":0.005,"watres":0.0},"output":', smp_out, '}'
  nrec = nrec + 1

  ! ================================================================
  ! 10. soil_hydraulic_conductivity(vol_liq, spafhy_para, khydr)
  ! ================================================================
  do k = 1, NTHETA
    theta = theta_grid(k)
    call soil_hydraulic_conductivity(theta, soil_params, khydr_out)
    write(10, '(A,F8.4,A,ES22.15,A)') &
      '{"fn":"soil_hydraulic_conductivity","inputs":{"vol_liq":', theta, &
      ',"n_van":1.12,"alpha_van":4.45,"watsat":0.75,"watres":0.0,"ksat":1.0e-5},"output":', khydr_out, '}'
    nrec = nrec + 1
  end do

  call soil_hydraulic_conductivity(0.004d0, soil_floor_params, khydr_out)
  write(10, '(A,ES22.15,A)') &
    '{"fn":"soil_hydraulic_conductivity","inputs":{"vol_liq":0.004,' // &
    '"n_van":1.12,"alpha_van":4.45,"watsat":0.005,"watres":0.0,"ksat":1.0e-5},"output":', khydr_out, '}'
  nrec = nrec + 1

  ! ================================================================
  ! 11. scale_conductivity(K, par_env)
  ! ================================================================
  do i = 1, NTC
    tc = tc_grid(i)
    do j = 1, NP
      patm = patm_grid(j)
      par_env%viscosity_water = viscosity_h2o(tc, patm)
      par_env%density_water   = density_h2o(tc, patm)
      par_env%patm            = patm
      par_env%tc              = tc
      par_env%vpd             = 1000.0d0
      result_d = scale_conductivity(4.0d-16, par_env)
      write(10, '(A,F10.4,A,F12.1,A,ES22.15,A)') &
        '{"fn":"scale_conductivity","inputs":{"K":4.0e-16,"tc":', tc, &
        ',"patm":', patm, '},"output":', result_d, '}'
      nrec = nrec + 1
    end do
  end do

  ! ================================================================
  ! 12. calc_gs(dpsi, psi_soil, par_plant, par_env)
  ! ================================================================
  par_plant%conductivity = 4.0d-16
  par_plant%psi50        = -3.46d0
  par_plant%b            = 2.0d0

  do i = 1, NTC
    tc = tc_grid(i)
    do j = 1, NP
      patm = patm_grid(j)
      par_env%viscosity_water = viscosity_h2o(tc, patm)
      par_env%density_water   = density_h2o(tc, patm)
      par_env%patm            = patm
      par_env%tc              = tc
      par_env%vpd             = 1000.0d0
      result_d = calc_gs(1.0d0, -0.5d0, par_plant, par_env)
      write(10, '(A,F10.4,A,F12.1,A,ES22.15,A)') &
        '{"fn":"calc_gs","inputs":{"dpsi":1.0,"psi_soil":-0.5,"tc":', tc, &
        ',"patm":', patm, &
        ',"vpd":1000.0,"conductivity":4.0e-16,"psi50":-3.46,"b":2.0},"output":', result_d, '}'
      nrec = nrec + 1
    end do
  end do

  ! ================================================================
  ! 13. calc_assim_light_limited(ci, aj, gs, jmax, par_photosynth)
  ! ================================================================
  do i = 1, NTC
    tc = tc_grid(i)
    do j = 1, NP
      patm = patm_grid(j)
      par_env%viscosity_water = viscosity_h2o(tc, patm)
      par_env%density_water   = density_h2o(tc, patm)
      par_env%patm            = patm
      par_env%tc              = tc
      par_env%vpd             = 1000.0d0

      call calc_kmm(tc, patm, par_photosynth%kmm)
      par_photosynth%gammastar = gammastar(tc, patm)
      par_photosynth%phi0      = kphio * ftemp_kphio(tc, .FALSE.)
      par_photosynth%Iabs      = 300.0d0
      par_photosynth%ca        = 400.0d0 * patm * 1.0d-6
      par_photosynth%patm      = patm
      par_photosynth%delta     = 0.015d0

      result_d = calc_gs(1.0d0, -0.5d0, par_plant, par_env)
      call calc_assim_light_limited(ci_out, aj_out, result_d, 100.0d0, par_photosynth)

      write(10, '(A,F10.4,A,F12.1,A,ES22.15,A,ES22.15,A)') &
        '{"fn":"calc_assim_light_limited","inputs":{"tc":', tc, &
        ',"patm":', patm, &
        ',"gs_from_calc_gs":true,"jmax":100.0,"Iabs":300.0,"ca_ppm":400.0,' // &
        '"delta":0.015},"output":{"ci":', ci_out, ',"aj":', aj_out, '}}'
      nrec = nrec + 1
    end do
  end do

  ! ================================================================
  ! 14. fn_profit(par, psi_soil, par_cost, par_photosynth, par_plant, par_env, .FALSE.)
  ! ================================================================
  par_cost%alpha = 0.1d0
  par_cost%gamma = 0.5d0

  do i = 1, NTC
    tc = tc_grid(i)
    do j = 1, NP
      patm = patm_grid(j)
      par_env%viscosity_water = viscosity_h2o(tc, patm)
      par_env%density_water   = density_h2o(tc, patm)
      par_env%patm            = patm
      par_env%tc              = tc
      par_env%vpd             = 1000.0d0

      call calc_kmm(tc, patm, par_photosynth%kmm)
      par_photosynth%gammastar = gammastar(tc, patm)
      par_photosynth%phi0      = kphio * ftemp_kphio(tc, .FALSE.)
      par_photosynth%Iabs      = 300.0d0
      par_photosynth%ca        = 400.0d0 * patm * 1.0d-6
      par_photosynth%patm      = patm
      par_photosynth%delta     = 0.015d0

      lj_dps%logjmax = 3.0d0
      lj_dps%dpsi    = 1.0d0

      result_d = fn_profit(lj_dps, -0.5d0, par_cost, par_photosynth, &
                           par_plant, par_env, .FALSE.)

      write(10, '(A,F10.4,A,F12.1,A,ES22.15,A)') &
        '{"fn":"fn_profit","inputs":{"logjmax":3.0,"dpsi":1.0,' // &
        '"psi_soil":-0.5,"alpha":0.1,"gamma":0.5,"tc":', tc, &
        ',"patm":', patm, &
        ',"vpd":1000.0,"Iabs":300.0,"ca_ppm":400.0,"delta":0.015,' // &
        '"conductivity":4.0e-16,"psi50":-3.46,"b":2.0,' // &
        '"hypothesis":"PM","do_optim":false},"output":', result_d, '}'
      nrec = nrec + 1
    end do
  end do

  tc = 20.0d0
  patm = 101325.0d0
  opt_hypothesis = 'LC'
  par_env%viscosity_water = viscosity_h2o(tc, patm)
  par_env%density_water   = density_h2o(tc, patm)
  par_env%patm            = patm
  par_env%tc              = tc
  par_env%vpd             = 1000.0d0

  call calc_kmm(tc, patm, par_photosynth%kmm)
  par_photosynth%gammastar = gammastar(tc, patm)
  par_photosynth%phi0      = kphio * ftemp_kphio(tc, .FALSE.)
  par_photosynth%Iabs      = 300.0d0
  par_photosynth%ca        = 400.0d0 * patm * 1.0d-6
  par_photosynth%patm      = patm
  par_photosynth%delta     = 0.015d0

  lj_dps%logjmax = 3.0d0
  lj_dps%dpsi    = 1.0d0
  result_d = fn_profit(lj_dps, -0.5d0, par_cost, par_photosynth, &
                       par_plant, par_env, .TRUE.)

  write(10, '(A,ES22.15,A)') &
    '{"fn":"fn_profit","inputs":{"logjmax":3.0,"dpsi":1.0,' // &
    '"psi_soil":-0.5,"alpha":0.1,"gamma":0.5,"tc":20.0,' // &
    '"patm":101325.0,"vpd":1000.0,"Iabs":300.0,"ca_ppm":400.0,' // &
    '"delta":0.015,"conductivity":4.0e-16,"psi50":-3.46,"b":2.0,' // &
    '"hypothesis":"LC","do_optim":true},"output":', result_d, '}'
  nrec = nrec + 1
  opt_hypothesis = 'PM'

  ! ================================================================
  ! 15. quadratic(a, b, c) → r1
  ! ================================================================
  do i = 1, NQUAD
    call quadratic(quad_a(i), quad_b(i), quad_c(i), quad_r1)
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)') &
      '{"fn":"quadratic","inputs":{"a":', quad_a(i), ',"b":', quad_b(i), &
      ',"c":', quad_c(i), '},"output":', quad_r1, '}'
    nrec = nrec + 1
  end do

  ! ================================================================
  ! 16. inputs_to_fractions(leaf, root, soluble, compost) → fract(5)
  ! ================================================================
  ! 5 test cases with different litter composition
  call inputs_to_fractions(1.0d0, 0.0d0, 0.0d0, 0.0d0, fract_out)
  write(10, '(A,5(ES22.15,A))') &
    '{"fn":"inputs_to_fractions","inputs":{"leaf":1.0,"root":0.0,"soluble":0.0,"compost":0.0},"output":[', &
    fract_out(1), ',', fract_out(2), ',', fract_out(3), ',', fract_out(4), ',', fract_out(5), ']}'
  nrec = nrec + 1

  call inputs_to_fractions(0.0d0, 1.0d0, 0.0d0, 0.0d0, fract_out)
  write(10, '(A,5(ES22.15,A))') &
    '{"fn":"inputs_to_fractions","inputs":{"leaf":0.0,"root":1.0,"soluble":0.0,"compost":0.0},"output":[', &
    fract_out(1), ',', fract_out(2), ',', fract_out(3), ',', fract_out(4), ',', fract_out(5), ']}'
  nrec = nrec + 1

  call inputs_to_fractions(0.0d0, 0.0d0, 1.0d0, 0.0d0, fract_out)
  write(10, '(A,5(ES22.15,A))') &
    '{"fn":"inputs_to_fractions","inputs":{"leaf":0.0,"root":0.0,"soluble":1.0,"compost":0.0},"output":[', &
    fract_out(1), ',', fract_out(2), ',', fract_out(3), ',', fract_out(4), ',', fract_out(5), ']}'
  nrec = nrec + 1

  call inputs_to_fractions(0.0d0, 0.0d0, 0.0d0, 1.0d0, fract_out)
  write(10, '(A,5(ES22.15,A))') &
    '{"fn":"inputs_to_fractions","inputs":{"leaf":0.0,"root":0.0,"soluble":0.0,"compost":1.0},"output":[', &
    fract_out(1), ',', fract_out(2), ',', fract_out(3), ',', fract_out(4), ',', fract_out(5), ']}'
  nrec = nrec + 1

  call inputs_to_fractions(0.5d0, 0.3d0, 0.1d0, 0.1d0, fract_out)
  write(10, '(A,5(ES22.15,A))') &
    '{"fn":"inputs_to_fractions","inputs":{"leaf":0.5,"root":0.3,"soluble":0.1,"compost":0.1},"output":[', &
    fract_out(1), ',', fract_out(2), ',', fract_out(3), ',', fract_out(4), ',', fract_out(5), ']}'
  nrec = nrec + 1

  ! ================================================================
  ! 17. aerodynamics(LAI, Uo, spafhy_para) — PRIVATE, duplicated here
  ! ================================================================
  do m = 1, NPARA_AERO
    do i = 1, NLAI_AERO
      do j = 1, NUO_AERO
        call harness_aerodynamics(lai_aero_grid(i), uo_aero_grid(j), &
          aero_ra, aero_rb, aero_ras, aero_ustar, aero_Uh, aero_Ug, aero_para(m))
        write(10, '(A,F6.1,A,F6.1,A,F6.1,A,F6.1,A,F6.2,A,F6.2,A,F6.3,A,' // &
          'ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)') &
          '{"fn":"aerodynamics","inputs":{"LAI":', lai_aero_grid(i), &
          ',"Uo":', uo_aero_grid(j), &
          ',"hc":', aero_para(m)%hc, &
          ',"zmeas":', aero_para(m)%zmeas, &
          ',"zground":', aero_para(m)%zground, &
          ',"zo_ground":', aero_para(m)%zo_ground, &
          ',"w_leaf":', aero_para(m)%w_leaf, &
          '},"output":{"ra":', aero_ra, &
          ',"rb":', aero_rb, &
          ',"ras":', aero_ras, &
          ',"ustar":', aero_ustar, &
          ',"Uh":', aero_Uh, &
          ',"Ug":', aero_Ug, '}}'
        nrec = nrec + 1
      end do
    end do
  end do

  call harness_aerodynamics(2.0d0, 2.0d0, aero_ra, aero_rb, aero_ras, aero_ustar, aero_Uh, aero_Ug, aero_cap_params)
  write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)') &
    '{"fn":"aerodynamics","inputs":{"LAI":2.0,"Uo":2.0,"hc":10.0,' // &
    '"zmeas":2.0,"zground":2.0,"zo_ground":0.02,"w_leaf":0.02},"output":{"ra":', aero_ra, &
    ',"rb":', aero_rb, ',"ras":', aero_ras, ',"ustar":', aero_ustar, &
    ',"Uh":', aero_Uh, ',"Ug":', aero_Ug, '}}'
  nrec = nrec + 1

  ! ================================================================
  ! 18. exponential_smooth_met — 3 steps showing initialization + smoothing
  ! ================================================================
  met_daily(1) = 10.0d0
  met_daily(2) = 0.5d0
  met_ind = 1

  ! Step 1: initialization (met_ind == 1) → rolling = daily
  call exponential_smooth_met(met_daily, met_rolling, met_ind)
  write(10, '(A,ES22.15,A,ES22.15,A,I1,A)') &
    '{"fn":"exponential_smooth_met","inputs":{"met_daily":[10.0,0.5],' // &
    '"met_rolling_in":[0.0,0.0],"met_ind_in":1},"output":{"met_rolling":[', &
    met_rolling(1), ',', met_rolling(2), '],"met_ind":', met_ind, '}}'
  nrec = nrec + 1

  ! Step 2: exponential smoothing with same daily
  call exponential_smooth_met(met_daily, met_rolling, met_ind)
  write(10, '(A,ES22.15,A,ES22.15,A,I1,A)') &
    '{"fn":"exponential_smooth_met","inputs":{"met_daily":[10.0,0.5],' // &
    '"met_rolling_in":[10.0,0.5],"met_ind_in":2},"output":{"met_rolling":[', &
    met_rolling(1), ',', met_rolling(2), '],"met_ind":', met_ind, '}}'
  nrec = nrec + 1

  ! Step 3: exponential smoothing with different daily
  met_daily(1) = 15.0d0
  met_daily(2) = 0.8d0
  call exponential_smooth_met(met_daily, met_rolling, met_ind)
  write(10, '(A,ES22.15,A,ES22.15,A,I1,A)') &
    '{"fn":"exponential_smooth_met","inputs":{"met_daily":[15.0,0.8],' // &
    '"met_rolling_in":[10.0,0.5],"met_ind_in":2},"output":{"met_rolling":[', &
    met_rolling(1), ',', met_rolling(2), '],"met_ind":', met_ind, '}}'
  nrec = nrec + 1

  ! ================================================================
  ! 19. pmodel_hydraulics_numerical (full P-Hydro solver)
  ! ================================================================
  ! Set module-level globals read by pmodel_hydraulics_numerical
  conductivity = 4.0d-16
  psi50        = -3.46d0
  b            = 2.0d0
  alpha        = 0.1d0
  gamma        = 0.5d0
  ! kphio already defaults to 0.087182 via readvegpara_mod
  opt_hypothesis = 'PM'

  ! Representative test cases spanning environmental gradients:
  !   tc(°C)  vpd(Pa)  ppfd    co2(ppm)  psi_soil(MPa)  fapar   sp(Pa)      rdark
  phn_tc       = (/ 10.0d0,  20.0d0,   25.0d0,   30.0d0,    15.0d0,     20.0d0  /)
  phn_vpd      = (/ 500.0d0, 1000.0d0, 2000.0d0, 3000.0d0,  800.0d0,    1500.0d0 /)
  phn_ppfd     = (/ 200.0d0, 300.0d0,  500.0d0,  400.0d0,   150.0d0,    350.0d0  /)
  phn_co2      = (/ 400.0d0, 400.0d0,  400.0d0,  400.0d0,   280.0d0,    600.0d0  /)
  phn_psi_soil = (/ -0.5d0,  -0.5d0,   -1.0d0,   -2.0d0,    -0.2d0,     -0.5d0  /)
  phn_fapar    = (/ 0.8d0,   0.9d0,    0.7d0,    0.6d0,     0.95d0,     0.85d0   /)
  phn_sp       = (/ 101325.0d0, 101325.0d0, 80000.0d0, 101325.0d0, 101325.0d0, 110000.0d0 /)
  phn_rdark    = (/ 0.015d0, 0.015d0,  0.015d0,  0.015d0,   0.015d0,    0.015d0  /)

  do i = 1, NPHN
    call pmodel_hydraulics_numerical( &
      phn_tc(i), phn_ppfd(i), phn_vpd(i), phn_co2(i), phn_sp(i), &
      phn_fapar(i), phn_psi_soil(i), phn_rdark(i), &
      phn_jmax, phn_dpsi, phn_gs, phn_aj, phn_ci, phn_chi, &
      phn_vcmax, phn_profit, phn_chi_jmax_lim)
    write(10, '(A,F10.4,A,F10.4,A,F10.4,A,F10.4,A,F12.1,A,' // &
      'F6.4,A,F8.4,A,F8.6,A,' // &
      'ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A,' // &
      'ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)') &
      '{"fn":"pmodel_hydraulics_numerical","inputs":{' // &
      '"tc":', phn_tc(i), &
      ',"ppfd":', phn_ppfd(i), &
      ',"vpd":', phn_vpd(i), &
      ',"co2":', phn_co2(i), &
      ',"sp":', phn_sp(i), &
      ',"fapar":', phn_fapar(i), &
      ',"psi_soil":', phn_psi_soil(i), &
      ',"rdark_leaf":', phn_rdark(i), &
      '},"output":{"jmax":', phn_jmax, &
      ',"dpsi":', phn_dpsi, &
      ',"gs":', phn_gs, &
      ',"aj":', phn_aj, &
      ',"ci":', phn_ci, &
      ',"chi":', phn_chi, &
      ',"vcmax":', phn_vcmax, &
      ',"profit":', phn_profit, &
      ',"chi_jmax_lim":', phn_chi_jmax_lim, '}}'
    nrec = nrec + 1
  end do

  ! ================================================================
  ! 20. canopy_water_snow (canopy interception + snowpack dynamics)
  ! ================================================================
  do i = 1, NCWF
    ! Reset canopy state from test case arrays
    cw_state%CanopyStorage = cwf_CanSto0(i)
    cw_state%SWE           = cwf_SWE0(i)
    cw_state%swe_i         = cwf_swe_i0(i)
    cw_state%swe_l         = cwf_swe_l0(i)
    ! Zero flux struct
    cw_flux%Throughfall     = 0.0d0
    cw_flux%Interception    = 0.0d0
    cw_flux%CanopyEvap      = 0.0d0
    cw_flux%Unloading       = 0.0d0
    cw_flux%SoilEvap        = 0.0d0
    cw_flux%ET              = 0.0d0
    cw_flux%Transpiration   = 0.0d0
    cw_flux%PotInfiltration = 0.0d0
    cw_flux%Melt            = 0.0d0
    cw_flux%Freeze          = 0.0d0
    cw_flux%mbe             = 0.0d0
    ! Compute Ra via aerodynamics
    call harness_aerodynamics(cwf_LAI(i), cwf_U(i), aero_ra, aero_rb, &
                              aero_ras, aero_ustar, aero_Uh, aero_Ug, cwf_para)
    ! Call canopy_water_snow (private -> harness duplicate)
    call harness_canopy_water_snow(cw_state, cw_flux, cwf_para, &
      cwf_Ta(i), cwf_Prec(i), cwf_Rn(i)*cwf_fapar(i), cwf_VPD(i), &
      aero_ra, cwf_U(i), cwf_LAI(i), cwf_P(i))
    ! Emit JSONL
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '{"fn":"canopy_water_snow","inputs":{"T":', cwf_Ta(i), &
      ',"Pre":', cwf_Prec(i), ',"AE":', cwf_Rn(i)*cwf_fapar(i), &
      ',"D":', cwf_VPD(i), ',"Ra":', aero_ra, ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"U":', cwf_U(i), ',"LAI":', cwf_LAI(i), ',"P":', cwf_P(i), ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"CanopyStorage_in":', cwf_CanSto0(i), ',"SWE_in":', cwf_SWE0(i), &
      ',"swe_i_in":', cwf_swe_i0(i), ',"swe_l_in":', cwf_swe_l0(i), ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"wmax":', cwf_para%wmax, ',"wmaxsnow":', cwf_para%wmaxsnow, &
      ',"kmelt":', cwf_para%kmelt, ',"kfreeze":', cwf_para%kfreeze, &
      ',"frac_snowliq":', cwf_para%frac_snowliq, ',"time_step":', time_step, '},'
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"output":{"CanopyStorage":', cw_state%CanopyStorage, &
      ',"SWE":', cw_state%SWE, ',"swe_i":', cw_state%swe_i, &
      ',"swe_l":', cw_state%swe_l, ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"Throughfall":', cw_flux%Throughfall, ',"Interception":', cw_flux%Interception, &
      ',"CanopyEvap":', cw_flux%CanopyEvap, ',"Unloading":', cw_flux%Unloading, ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)') &
      '"PotInfiltration":', cw_flux%PotInfiltration, ',"Melt":', cw_flux%Melt, &
      ',"Freeze":', cw_flux%Freeze, ',"mbe":', cw_flux%mbe, '}}'
    nrec = nrec + 1
  end do

  ! ================================================================
  ! 21. ground_evaporation (soil surface evaporation)
  ! ================================================================
  do i = 1, NCWF
    ! Reset canopy state (for SWE check)
    cw_state%CanopyStorage = cwf_CanSto0(i)
    cw_state%SWE           = cwf_SWE0(i)
    cw_state%swe_i         = cwf_swe_i0(i)
    cw_state%swe_l         = cwf_swe_l0(i)
    ! Zero flux
    cw_flux%SoilEvap = 0.0d0
    ! Set soil state
    sw_state%WatSto = cwf_WatSto0(i)
    sw_state%beta   = cwf_beta0(i)
    ! Compute Ras via aerodynamics
    call harness_aerodynamics(cwf_LAI(i), cwf_U(i), aero_ra, aero_rb, &
                              aero_ras, aero_ustar, aero_Uh, aero_Ug, cwf_para)
    ! Call ground_evaporation (private -> harness duplicate)
    call harness_ground_evaporation(cw_state, cw_flux, sw_state, cwf_para, &
      cwf_Ta(i), cwf_Rn(i)*(1.0d0-cwf_fapar(i)), cwf_VPD(i), aero_ras, cwf_P(i))
    ! Emit JSONL
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '{"fn":"ground_evaporation","inputs":{"T":', cwf_Ta(i), &
      ',"AE":', cwf_Rn(i)*(1.0d0-cwf_fapar(i)), ',"VPD":', cwf_VPD(i), &
      ',"Ras":', aero_ras, ',"P":', cwf_P(i), ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"SWE":', cwf_SWE0(i), ',"beta":', cwf_beta0(i), &
      ',"WatSto":', cwf_WatSto0(i), ',"gsoil":', cwf_para%gsoil, ','
    write(10, '(A,ES22.15,A,ES22.15,A)') &
      '"time_step":', time_step, '},"output":{"SoilEvap":', cw_flux%SoilEvap, '}}'
    nrec = nrec + 1
  end do

  ! ================================================================
  ! 22. canopy_water_flux (orchestrator: aerodynamics + canopy_water_snow + ground_evaporation)
  ! ================================================================
  do i = 1, NCWF
    ! Reset canopy state from test case arrays
    cw_state%CanopyStorage = cwf_CanSto0(i)
    cw_state%SWE           = cwf_SWE0(i)
    cw_state%swe_i         = cwf_swe_i0(i)
    cw_state%swe_l         = cwf_swe_l0(i)
    ! Zero flux struct
    cw_flux%Throughfall     = 0.0d0
    cw_flux%Interception    = 0.0d0
    cw_flux%CanopyEvap      = 0.0d0
    cw_flux%Unloading       = 0.0d0
    cw_flux%SoilEvap        = 0.0d0
    cw_flux%ET              = 0.0d0
    cw_flux%Transpiration   = 0.0d0
    cw_flux%PotInfiltration = 0.0d0
    cw_flux%Melt            = 0.0d0
    cw_flux%Freeze          = 0.0d0
    cw_flux%mbe             = 0.0d0
    ! Set soil state for ground evaporation
    sw_state%WatSto = cwf_WatSto0(i)
    sw_state%beta   = cwf_beta0(i)
    ! Call public canopy_water_flux
    call canopy_water_flux(cwf_Rn(i), cwf_Ta(i), cwf_Prec(i), cwf_VPD(i), &
      cwf_U(i), cwf_P(i), cwf_fapar(i), cwf_LAI(i), &
      cw_state, cw_flux, sw_state, cwf_para)
    ! Emit JSONL - meteorological inputs
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '{"fn":"canopy_water_flux","inputs":{"Rn":', cwf_Rn(i), &
      ',"Ta":', cwf_Ta(i), ',"Prec":', cwf_Prec(i), ',"VPD":', cwf_VPD(i), ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"U":', cwf_U(i), ',"P":', cwf_P(i), &
      ',"fapar":', cwf_fapar(i), ',"LAI":', cwf_LAI(i), ','
    ! Input state
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"CanopyStorage_in":', cwf_CanSto0(i), ',"SWE_in":', cwf_SWE0(i), &
      ',"swe_i_in":', cwf_swe_i0(i), ',"swe_l_in":', cwf_swe_l0(i), ','
    write(10, '(A,ES22.15,A,ES22.15,A)', advance='no') &
      '"WatSto_in":', cwf_WatSto0(i), ',"beta_in":', cwf_beta0(i), ','
    ! Parameters
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"hc":', cwf_para%hc, ',"zmeas":', cwf_para%zmeas, &
      ',"zground":', cwf_para%zground, ',"zo_ground":', cwf_para%zo_ground, &
      ',"w_leaf":', cwf_para%w_leaf, ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"wmax":', cwf_para%wmax, ',"wmaxsnow":', cwf_para%wmaxsnow, &
      ',"gsoil":', cwf_para%gsoil, ',"kmelt":', cwf_para%kmelt, ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"kfreeze":', cwf_para%kfreeze, ',"frac_snowliq":', cwf_para%frac_snowliq, &
      ',"time_step":', time_step, '},'
    ! Output state
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"output":{"CanopyStorage":', cw_state%CanopyStorage, &
      ',"SWE":', cw_state%SWE, ',"swe_i":', cw_state%swe_i, &
      ',"swe_l":', cw_state%swe_l, ','
    ! Output flux
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"Throughfall":', cw_flux%Throughfall, ',"Interception":', cw_flux%Interception, &
      ',"CanopyEvap":', cw_flux%CanopyEvap, ',"Unloading":', cw_flux%Unloading, ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"SoilEvap":', cw_flux%SoilEvap, ',"ET":', cw_flux%ET, &
      ',"Transpiration":', cw_flux%Transpiration, ',"PotInfiltration":', cw_flux%PotInfiltration, ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A)') &
      '"Melt":', cw_flux%Melt, ',"Freeze":', cw_flux%Freeze, &
      ',"mbe":', cw_flux%mbe, '}}'
    nrec = nrec + 1
  end do

  ! ================================================================
  ! 23. soil_water (1-layer soil water balance)
  ! ================================================================
  do i = 1, NSW
    ! Reset soil state from test case arrays
    sw_state%WatSto     = sw_WatSto0(i)
    sw_state%PondSto    = sw_PondSto0(i)
    sw_state%MaxWatSto  = sw_MaxWatSto0(i)
    sw_state%MaxPondSto = sw_MaxPondSto0(i)
    sw_state%FcSto      = sw_FcSto0(i)
    sw_state%Wliq       = sw_Wliq0(i)
    sw_state%Kh         = sw_Kh0(i)
    sw_state%Psi        = 0.0d0
    sw_state%Sat        = 0.0d0
    sw_state%beta       = 0.0d0
    ! Zero flux struct
    sw_flux%Infiltration = 0.0d0
    sw_flux%Runoff       = 0.0d0
    sw_flux%Drainage     = 0.0d0
    sw_flux%LateralFlow  = 0.0d0
    sw_flux%ET           = 0.0d0
    sw_flux%mbe          = 0.0d0
    ! Copy inout args (soil_water modifies tr, evap, latflow)
    sw_tr_out      = sw_tr(i)
    sw_evap_out    = sw_evap(i)
    sw_latflow_out = sw_latflow(i)
    ! Call public soil_water
    call soil_water(sw_state, sw_flux, cwf_para, sw_potinf(i), &
      sw_tr_out, sw_evap_out, sw_latflow_out)
    ! Emit JSONL - inputs
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '{"fn":"soil_water","inputs":{"potinf":', sw_potinf(i), &
      ',"tr":', sw_tr(i), ',"evap":', sw_evap(i), &
      ',"latflow":', sw_latflow(i), ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"WatSto_in":', sw_WatSto0(i), ',"PondSto_in":', sw_PondSto0(i), &
      ',"MaxWatSto":', sw_MaxWatSto0(i), ',"MaxPondSto":', sw_MaxPondSto0(i), ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"FcSto":', sw_FcSto0(i), ',"Wliq_in":', sw_Wliq0(i), &
      ',"Kh_in":', sw_Kh0(i), ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"max_poros":', cwf_para%max_poros, ',"n_van":', cwf_para%n_van, &
      ',"alpha_van":', cwf_para%alpha_van, ',"watsat":', cwf_para%watsat, ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"watres":', cwf_para%watres, ',"ksat":', cwf_para%ksat, &
      ',"time_step":', time_step, '},'
    ! Output state
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"output":{"WatSto":', sw_state%WatSto, ',"PondSto":', sw_state%PondSto, &
      ',"Wliq":', sw_state%Wliq, ',"Sat":', sw_state%Sat, ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"beta":', sw_state%beta, ',"Psi":', sw_state%Psi, &
      ',"Kh":', sw_state%Kh, ','
    ! Output flux
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"Infiltration":', sw_flux%Infiltration, ',"Drainage":', sw_flux%Drainage, &
      ',"ET":', sw_flux%ET, ',"Runoff":', sw_flux%Runoff, ','
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      '"LateralFlow":', sw_flux%LateralFlow, ',"mbe":', sw_flux%mbe, &
      ',"tr_out":', sw_tr_out, ','
    write(10, '(A,ES22.15,A,ES22.15,A)') &
      '"evap_out":', sw_evap_out, ',"latflow_out":', sw_latflow_out, '}}'
    nrec = nrec + 1
  end do

  ! ================================================================
  ! 24. matrixnorm + matrixexp (yasso20 module)
  ! ================================================================
  ! Test matrix 1: zero matrix → exp(0) = I
  mexp_A = 0.0d0

  call matrixnorm(mexp_A, mnorm_p)
  write(10, '(A)', advance='no') '{"fn":"matrixnorm","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A,ES22.15,A)') '},"output":', mnorm_p, '}'
  nrec = nrec + 1

  call matrixexp(mexp_A, mexp_B)
  write(10, '(A)', advance='no') '{"fn":"matrixexp","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A)', advance='no') '},"output":'
  call write_mat5_json(10, mexp_B)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! Test matrix 2: diagonal with small negative values (typical decomposition rates)
  mexp_A = 0.0d0
  mexp_A(1,1) = -0.5d0
  mexp_A(2,2) = -0.3d0
  mexp_A(3,3) = -0.2d0
  mexp_A(4,4) = -0.1d0
  mexp_A(5,5) = -0.05d0

  call matrixnorm(mexp_A, mnorm_p)
  write(10, '(A)', advance='no') '{"fn":"matrixnorm","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A,ES22.15,A)') '},"output":', mnorm_p, '}'
  nrec = nrec + 1

  call matrixexp(mexp_A, mexp_B)
  write(10, '(A)', advance='no') '{"fn":"matrixexp","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A)', advance='no') '},"output":'
  call write_mat5_json(10, mexp_B)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! Test matrix 3: Yasso-like decomposition matrix
  ! Negative diagonal (decomposition rates), positive off-diagonal (transfers)
  mexp_A = 0.0d0
  mexp_A(1,1) = -0.48d0
  mexp_A(2,2) = -0.33d0
  mexp_A(3,3) = -0.22d0
  mexp_A(4,4) = -0.10d0
  mexp_A(5,5) = -0.008d0
  ! Off-diagonal transfers (A→W, W→E, E→N, A→H, etc.)
  mexp_A(2,1) =  0.10d0
  mexp_A(3,2) =  0.05d0
  mexp_A(4,3) =  0.03d0
  mexp_A(5,1) =  0.02d0
  mexp_A(5,2) =  0.01d0
  mexp_A(5,3) =  0.01d0
  mexp_A(5,4) =  0.005d0
  mexp_A(1,2) =  0.08d0
  mexp_A(3,1) =  0.04d0

  call matrixnorm(mexp_A, mnorm_p)
  write(10, '(A)', advance='no') '{"fn":"matrixnorm","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A,ES22.15,A)') '},"output":', mnorm_p, '}'
  nrec = nrec + 1

  call matrixexp(mexp_A, mexp_B)
  write(10, '(A)', advance='no') '{"fn":"matrixexp","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A)', advance='no') '},"output":'
  call write_mat5_json(10, mexp_B)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! Test matrix 4: larger-norm version (10× case 3) → exercises more scaling
  mexp_A = mexp_A * 10.0d0

  call matrixnorm(mexp_A, mnorm_p)
  write(10, '(A)', advance='no') '{"fn":"matrixnorm","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A,ES22.15,A)') '},"output":', mnorm_p, '}'
  nrec = nrec + 1

  call matrixexp(mexp_A, mexp_B)
  write(10, '(A)', advance='no') '{"fn":"matrixexp","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A)', advance='no') '},"output":'
  call write_mat5_json(10, mexp_B)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! Test matrix 5: identity matrix → exp(I)
  mexp_A = 0.0d0
  do i = 1, 5
    mexp_A(i,i) = 1.0d0
  end do

  call matrixnorm(mexp_A, mnorm_p)
  write(10, '(A)', advance='no') '{"fn":"matrixnorm","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A,ES22.15,A)') '},"output":', mnorm_p, '}'
  nrec = nrec + 1

  call matrixexp(mexp_A, mexp_B)
  write(10, '(A)', advance='no') '{"fn":"matrixexp","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A)', advance='no') '},"output":'
  call write_mat5_json(10, mexp_B)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! Test matrices 6-8: real mod5c20 coefficient-matrix path.
  ! These are generated by the same A-matrix construction used inside
  ! yasso20.mod5c20 before the matrix exponential is applied.

  ! Case 6: cool seasonal climate, no size effect, 1-year step.
  mexp_theta = 0.0d0
  mexp_theta(1:4) = (/ 0.70d0, 0.45d0, 0.25d0, 0.12d0 /)
  mexp_theta(5:16) = (/ 0.10d0, 0.05d0, 0.04d0, 0.08d0, 0.06d0, 0.03d0, &
                        0.04d0, 0.05d0, 0.02d0, 0.03d0, 0.02d0, 0.01d0 /)
  mexp_theta(22:30) = (/ 0.08d0, -0.0015d0, 0.06d0, -0.0012d0, 0.03d0, -0.0008d0, &
                         -0.50d0, -0.45d0, -0.35d0 /)
  mexp_theta(31:35) = (/ 0.04d0, 0.006d0, 0.00d0, 0.00d0, 0.00d0 /)
  mexp_temp = (/ -5.0d0, -3.0d0, 0.0d0, 5.0d0, 10.0d0, 14.0d0, &
                  16.0d0, 15.0d0, 10.0d0, 5.0d0, 0.0d0, -4.0d0 /)
  mexp_time = 1.0d0
  mexp_prec = 650.0d0
  mexp_d = 0.0d0
  mexp_leac = 0.0d0
  call build_mod5c20_matrix(mexp_theta, mexp_time, mexp_temp, mexp_prec, mexp_d, mexp_leac, mexp_A)

  call matrixnorm(mexp_A, mnorm_p)
  write(10, '(A)', advance='no') '{"fn":"matrixnorm","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A,ES22.15,A)') '},"output":', mnorm_p, '}'
  nrec = nrec + 1

  call matrixexp(mexp_A, mexp_B)
  write(10, '(A)', advance='no') '{"fn":"matrixexp","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A)', advance='no') '},"output":'
  call write_mat5_json(10, mexp_B)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! Case 7: warm wet climate, woody size effect, 4-year step.
  mexp_theta(31:35) = (/ 0.05d0, 0.010d0, 0.020d0, 0.0005d0, 0.80d0 /)
  mexp_temp = (/ 1.0d0, 2.0d0, 5.0d0, 9.0d0, 14.0d0, 18.0d0, &
                  21.0d0, 20.0d0, 15.0d0, 10.0d0, 5.0d0, 2.0d0 /)
  mexp_time = 4.0d0
  mexp_prec = 1200.0d0
  mexp_d = 12.0d0
  mexp_leac = 0.0005d0
  call build_mod5c20_matrix(mexp_theta, mexp_time, mexp_temp, mexp_prec, mexp_d, mexp_leac, mexp_A)

  call matrixnorm(mexp_A, mnorm_p)
  write(10, '(A)', advance='no') '{"fn":"matrixnorm","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A,ES22.15,A)') '},"output":', mnorm_p, '}'
  nrec = nrec + 1

  call matrixexp(mexp_A, mexp_B)
  write(10, '(A)', advance='no') '{"fn":"matrixexp","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A)', advance='no') '},"output":'
  call write_mat5_json(10, mexp_B)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! Case 8: colder climate with leaching and longer 6-year step.
  mexp_theta(31:35) = (/ 0.03d0, 0.008d0, 0.015d0, 0.0003d0, 0.60d0 /)
  mexp_temp = (/ -8.0d0, -6.0d0, -2.0d0, 2.0d0, 7.0d0, 11.0d0, &
                  14.0d0, 13.0d0, 8.0d0, 3.0d0, -1.0d0, -5.0d0 /)
  mexp_time = 6.0d0
  mexp_prec = 900.0d0
  mexp_d = 6.0d0
  mexp_leac = 0.0010d0
  call build_mod5c20_matrix(mexp_theta, mexp_time, mexp_temp, mexp_prec, mexp_d, mexp_leac, mexp_A)

  call matrixnorm(mexp_A, mnorm_p)
  write(10, '(A)', advance='no') '{"fn":"matrixnorm","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A,ES22.15,A)') '},"output":', mnorm_p, '}'
  nrec = nrec + 1

  call matrixexp(mexp_A, mexp_B)
  write(10, '(A)', advance='no') '{"fn":"matrixexp","inputs":{"a":'
  call write_mat5_json(10, mexp_A)
  write(10, '(A)', advance='no') '},"output":'
  call write_mat5_json(10, mexp_B)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! ================================================================
  ! 25. mod5c20 — full yearly Yasso20 ODE solver
  ! ================================================================
  ! Reuse theta/temp/prec/d/leac from matrixexp section (case 6 setup).
  ! Reset theta to case-6 values to have a clean starting point.
  mexp_theta = 0.0d0
  mexp_theta(1:4) = (/ 0.70d0, 0.45d0, 0.25d0, 0.12d0 /)
  mexp_theta(5:16) = (/ 0.10d0, 0.05d0, 0.04d0, 0.08d0, &
                        0.06d0, 0.03d0, 0.04d0, 0.05d0, &
                        0.02d0, 0.03d0, 0.02d0, 0.01d0 /)
  mexp_theta(22:30) = (/ 0.08d0, -0.0015d0, 0.06d0, -0.0012d0, &
                         0.03d0, -0.0008d0, &
                         -0.50d0, -0.45d0, -0.35d0 /)
  mexp_theta(31:35) = (/ 0.04d0, 0.006d0, 0.00d0, 0.00d0, 0.00d0 /)

  ! Case 1: Nordic climate, 1-year step, non-zero init + input.
  mexp_temp = (/ -5.0d0, -3.0d0, 0.0d0, 5.0d0, 10.0d0, 14.0d0, &
                  16.0d0, 15.0d0, 10.0d0, 5.0d0, 0.0d0, -4.0d0 /)
  mexp_time = 1.0d0
  mexp_prec = 650.0d0
  mexp_d = 0.0d0
  mexp_leac = 0.0d0
  m5c_init = (/ 2.0d0, 1.5d0, 1.0d0, 0.5d0, 5.0d0 /)
  m5c_b = (/ 0.10d0, 0.08d0, 0.05d0, 0.02d0, 0.0d0 /)
  call mod5c20(mexp_theta, mexp_time, mexp_temp, mexp_prec, &
               m5c_init, m5c_b, mexp_d, mexp_leac, m5c_xt)
  write(10, '(A)', advance='no') '{"fn":"mod5c20","inputs":{' // &
    '"theta":'
  call write_vec_json(10, mexp_theta, 35)
  write(10, '(A,ES22.15,A)', advance='no') ',"time":', mexp_time, ',"temp":'
  call write_vec_json(10, mexp_temp, 12)
  write(10, '(A,ES22.15,A)', advance='no') ',"prec":', mexp_prec, ',"init":'
  call write_vec_json(10, m5c_init, 5)
  write(10, '(A)', advance='no') ',"b":'
  call write_vec_json(10, m5c_b, 5)
  write(10, '(A,ES22.15,A,ES22.15)', advance='no') &
    ',"d":', mexp_d, ',"leac":', mexp_leac
  write(10, '(A)', advance='no') '},"output":'
  call write_vec_json(10, m5c_xt, 5)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! Case 2: Same Nordic climate, 10-year step (longer horizon).
  mexp_time = 10.0d0
  call mod5c20(mexp_theta, mexp_time, mexp_temp, mexp_prec, &
               m5c_init, m5c_b, mexp_d, mexp_leac, m5c_xt)
  write(10, '(A)', advance='no') '{"fn":"mod5c20","inputs":{' // &
    '"theta":'
  call write_vec_json(10, mexp_theta, 35)
  write(10, '(A,ES22.15,A)', advance='no') ',"time":', mexp_time, ',"temp":'
  call write_vec_json(10, mexp_temp, 12)
  write(10, '(A,ES22.15,A)', advance='no') ',"prec":', mexp_prec, ',"init":'
  call write_vec_json(10, m5c_init, 5)
  write(10, '(A)', advance='no') ',"b":'
  call write_vec_json(10, m5c_b, 5)
  write(10, '(A,ES22.15,A,ES22.15)', advance='no') &
    ',"d":', mexp_d, ',"leac":', mexp_leac
  write(10, '(A)', advance='no') '},"output":'
  call write_vec_json(10, m5c_xt, 5)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! Case 3: Warm climate, woody size effect, 4-year step, leaching.
  mexp_theta(31:35) = (/ 0.05d0, 0.010d0, 0.020d0, 0.0005d0, 0.80d0 /)
  mexp_temp = (/ 1.0d0, 2.0d0, 5.0d0, 9.0d0, 14.0d0, 18.0d0, &
                  21.0d0, 20.0d0, 15.0d0, 10.0d0, 5.0d0, 2.0d0 /)
  mexp_time = 4.0d0
  mexp_prec = 1200.0d0
  mexp_d = 12.0d0
  mexp_leac = 0.0005d0
  m5c_init = (/ 3.0d0, 2.0d0, 1.5d0, 0.8d0, 8.0d0 /)
  m5c_b = (/ 0.15d0, 0.12d0, 0.08d0, 0.03d0, 0.0d0 /)
  call mod5c20(mexp_theta, mexp_time, mexp_temp, mexp_prec, &
               m5c_init, m5c_b, mexp_d, mexp_leac, m5c_xt)
  write(10, '(A)', advance='no') '{"fn":"mod5c20","inputs":{' // &
    '"theta":'
  call write_vec_json(10, mexp_theta, 35)
  write(10, '(A,ES22.15,A)', advance='no') ',"time":', mexp_time, ',"temp":'
  call write_vec_json(10, mexp_temp, 12)
  write(10, '(A,ES22.15,A)', advance='no') ',"prec":', mexp_prec, ',"init":'
  call write_vec_json(10, m5c_init, 5)
  write(10, '(A)', advance='no') ',"b":'
  call write_vec_json(10, m5c_b, 5)
  write(10, '(A,ES22.15,A,ES22.15)', advance='no') &
    ',"d":', mexp_d, ',"leac":', mexp_leac
  write(10, '(A)', advance='no') '},"output":'
  call write_vec_json(10, m5c_xt, 5)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! Case 4: Zero initial state (input-driven growth only).
  ! Reset to case-1 parameters.
  mexp_theta(31:35) = (/ 0.04d0, 0.006d0, 0.00d0, 0.00d0, 0.00d0 /)
  mexp_temp = (/ -5.0d0, -3.0d0, 0.0d0, 5.0d0, 10.0d0, 14.0d0, &
                  16.0d0, 15.0d0, 10.0d0, 5.0d0, 0.0d0, -4.0d0 /)
  mexp_time = 1.0d0
  mexp_prec = 650.0d0
  mexp_d = 0.0d0
  mexp_leac = 0.0d0
  m5c_init = 0.0d0
  m5c_b = (/ 0.10d0, 0.08d0, 0.05d0, 0.02d0, 0.0d0 /)
  call mod5c20(mexp_theta, mexp_time, mexp_temp, mexp_prec, &
               m5c_init, m5c_b, mexp_d, mexp_leac, m5c_xt)
  write(10, '(A)', advance='no') '{"fn":"mod5c20","inputs":{' // &
    '"theta":'
  call write_vec_json(10, mexp_theta, 35)
  write(10, '(A,ES22.15,A)', advance='no') ',"time":', mexp_time, ',"temp":'
  call write_vec_json(10, mexp_temp, 12)
  write(10, '(A,ES22.15,A)', advance='no') ',"prec":', mexp_prec, ',"init":'
  call write_vec_json(10, m5c_init, 5)
  write(10, '(A)', advance='no') ',"b":'
  call write_vec_json(10, m5c_b, 5)
  write(10, '(A,ES22.15,A,ES22.15)', advance='no') &
    ',"d":', mexp_d, ',"leac":', mexp_leac
  write(10, '(A)', advance='no') '},"output":'
  call write_vec_json(10, m5c_xt, 5)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! Case 5: Zero input b=0 (pure decay, no litter input).
  m5c_init = (/ 2.0d0, 1.5d0, 1.0d0, 0.5d0, 5.0d0 /)
  m5c_b = 0.0d0
  call mod5c20(mexp_theta, mexp_time, mexp_temp, mexp_prec, &
               m5c_init, m5c_b, mexp_d, mexp_leac, m5c_xt)
  write(10, '(A)', advance='no') '{"fn":"mod5c20","inputs":{' // &
    '"theta":'
  call write_vec_json(10, mexp_theta, 35)
  write(10, '(A,ES22.15,A)', advance='no') ',"time":', mexp_time, ',"temp":'
  call write_vec_json(10, mexp_temp, 12)
  write(10, '(A,ES22.15,A)', advance='no') ',"prec":', mexp_prec, ',"init":'
  call write_vec_json(10, m5c_init, 5)
  write(10, '(A)', advance='no') ',"b":'
  call write_vec_json(10, m5c_b, 5)
  write(10, '(A,ES22.15,A,ES22.15)', advance='no') &
    ',"d":', mexp_d, ',"leac":', mexp_leac
  write(10, '(A)', advance='no') '},"output":'
  call write_vec_json(10, m5c_xt, 5)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! Case 6: Steady-state prediction (solve A*x = -b for equilibrium).
  m5c_b = (/ 0.10d0, 0.08d0, 0.05d0, 0.02d0, 0.0d0 /)
  call mod5c20(mexp_theta, mexp_time, mexp_temp, mexp_prec, &
               m5c_init, m5c_b, mexp_d, mexp_leac, m5c_xt, &
               steadystate_pred=.true.)
  write(10, '(A)', advance='no') '{"fn":"mod5c20","inputs":{' // &
    '"theta":'
  call write_vec_json(10, mexp_theta, 35)
  write(10, '(A,ES22.15,A)', advance='no') ',"time":', mexp_time, ',"temp":'
  call write_vec_json(10, mexp_temp, 12)
  write(10, '(A,ES22.15,A)', advance='no') ',"prec":', mexp_prec, ',"init":'
  call write_vec_json(10, m5c_init, 5)
  write(10, '(A)', advance='no') ',"b":'
  call write_vec_json(10, m5c_b, 5)
  write(10, '(A,ES22.15,A,ES22.15)', advance='no') &
    ',"d":', mexp_d, ',"leac":', mexp_leac
  write(10, '(A)', advance='no') ',"steadystate_pred":true},"output":'
  call write_vec_json(10, m5c_xt, 5)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! Case 7: Very cold climate — tem ≈ 3e-8 (> tol) so the transient
  ! path runs, but decomposition is negligible (result ≈ init + b·time).
  mexp_temp = (/ -80.0d0, -80.0d0, -80.0d0, -80.0d0, -80.0d0, -80.0d0, &
                  -80.0d0, -80.0d0, -80.0d0, -80.0d0, -80.0d0, -80.0d0 /)
  m5c_init = (/ 2.0d0, 1.5d0, 1.0d0, 0.5d0, 5.0d0 /)
  m5c_b = (/ 0.10d0, 0.08d0, 0.05d0, 0.02d0, 0.0d0 /)
  call mod5c20(mexp_theta, mexp_time, mexp_temp, mexp_prec, &
               m5c_init, m5c_b, mexp_d, mexp_leac, m5c_xt)
  write(10, '(A)', advance='no') '{"fn":"mod5c20","inputs":{' // &
    '"theta":'
  call write_vec_json(10, mexp_theta, 35)
  write(10, '(A,ES22.15,A)', advance='no') ',"time":', mexp_time, ',"temp":'
  call write_vec_json(10, mexp_temp, 12)
  write(10, '(A,ES22.15,A)', advance='no') ',"prec":', mexp_prec, ',"init":'
  call write_vec_json(10, m5c_init, 5)
  write(10, '(A)', advance='no') ',"b":'
  call write_vec_json(10, m5c_b, 5)
  write(10, '(A,ES22.15,A,ES22.15)', advance='no') &
    ',"d":', mexp_d, ',"leac":', mexp_leac
  write(10, '(A)', advance='no') '},"output":'
  call write_vec_json(10, m5c_xt, 5)
  write(10, '(A)') '}'
  nrec = nrec + 1

  ! ================================================================
  ! 26a. initialize_totc — Yasso C/N pool initialization
  ! ================================================================
  ! All cases use Yasso20 MAP parameters

  ! Case 1: Pure equilibrium (no legacy), temperate baseline
  itc_totc(1) = 10.0d0;     itc_cn_input(1) = 20.0d0
  itc_fract_root(1) = 0.5d0; itc_fract_legacy(1) = 0.0d0
  itc_tempr_c(1) = 10.0d0;  itc_precip_day(1) = 2.0d0; itc_tempr_ampl(1) = 10.0d0

  ! Case 2: Full legacy (all carbon in H pool)
  itc_totc(2) = 10.0d0;     itc_cn_input(2) = 20.0d0
  itc_fract_root(2) = 0.5d0; itc_fract_legacy(2) = 1.0d0
  itc_tempr_c(2) = 10.0d0;  itc_precip_day(2) = 2.0d0; itc_tempr_ampl(2) = 10.0d0

  ! Case 3: Half legacy blend
  itc_totc(3) = 10.0d0;     itc_cn_input(3) = 20.0d0
  itc_fract_root(3) = 0.5d0; itc_fract_legacy(3) = 0.5d0
  itc_tempr_c(3) = 10.0d0;  itc_precip_day(3) = 2.0d0; itc_tempr_ampl(3) = 10.0d0

  ! Case 4: All root input
  itc_totc(4) = 10.0d0;     itc_cn_input(4) = 20.0d0
  itc_fract_root(4) = 1.0d0; itc_fract_legacy(4) = 0.0d0
  itc_tempr_c(4) = 10.0d0;  itc_precip_day(4) = 2.0d0; itc_tempr_ampl(4) = 10.0d0

  ! Case 5: All leaf input (metamorphic: should match case 4, since awenh_leaf == awenh_fineroot)
  itc_totc(5) = 10.0d0;     itc_cn_input(5) = 20.0d0
  itc_fract_root(5) = 0.0d0; itc_fract_legacy(5) = 0.0d0
  itc_tempr_c(5) = 10.0d0;  itc_precip_day(5) = 2.0d0; itc_tempr_ampl(5) = 10.0d0

  ! Case 6: Cold climate with large amplitude
  itc_totc(6) = 10.0d0;     itc_cn_input(6) = 20.0d0
  itc_fract_root(6) = 0.5d0; itc_fract_legacy(6) = 0.0d0
  itc_tempr_c(6) = -5.0d0;  itc_precip_day(6) = 1.0d0; itc_tempr_ampl(6) = 20.0d0

  ! Case 7: High total carbon
  itc_totc(7) = 100.0d0;    itc_cn_input(7) = 20.0d0
  itc_fract_root(7) = 0.5d0; itc_fract_legacy(7) = 0.0d0
  itc_tempr_c(7) = 10.0d0;  itc_precip_day(7) = 2.0d0; itc_tempr_ampl(7) = 10.0d0

  ! Case 8: Low total carbon
  itc_totc(8) = 0.1d0;      itc_cn_input(8) = 15.0d0
  itc_fract_root(8) = 0.3d0; itc_fract_legacy(8) = 0.2d0
  itc_tempr_c(8) = 15.0d0;  itc_precip_day(8) = 3.0d0; itc_tempr_ampl(8) = 5.0d0

  do i = 1, NINIT_TOTC
    call initialize_totc(param_y20_map, itc_totc(i), itc_cn_input(i), &
                         itc_fract_root(i), itc_fract_legacy(i), &
                         itc_tempr_c(i), itc_precip_day(i), itc_tempr_ampl(i), &
                         itc_cstate_out, itc_nstate_out)
    write(10, '(A)', advance='no') '{"fn":"initialize_totc","inputs":{"param":'
    call write_vec_json(10, param_y20_map, 35)
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"totc":', itc_totc(i), &
      ',"cn_input":', itc_cn_input(i), &
      ',"fract_root_input":', itc_fract_root(i)
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"fract_legacy_soc":', itc_fract_legacy(i), &
      ',"tempr_c":', itc_tempr_c(i), &
      ',"precip_day":', itc_precip_day(i), &
      ',"tempr_ampl":', itc_tempr_ampl(i)
    write(10, '(A)', advance='no') '},"output":{"cstate":'
    call write_vec_json(10, itc_cstate_out, 5)
    write(10, '(A,ES22.15,A)') ',"nstate":', itc_nstate_out, '}}'
    nrec = nrec + 1
  end do

  ! ================================================================
  ! 26. decompose — daily C/N decomposition step
  ! ================================================================
  dec_param = param_y20_map  ! use the Yasso20 MAP parameters
  dec_timestep_days = 1.0d0

  ! Test cases: varied temperature and precipitation
  dec_tempr_c = (/ 10.0d0, 20.0d0, 0.0d0, -10.0d0, 30.0d0, &
                    10.0d0, 10.0d0, 10.0d0, 10.0d0, 10.0d0 /)
  dec_precip_day = (/ 2.0d0, 2.0d0, 2.0d0, 2.0d0, 2.0d0, &
                      0.5d0, 5.0d0, 2.0d0, 2.0d0, 2.0d0 /)

  ! Carbon states for testing (5 AWENH pools)
  ! Cases 1-5: same C state, varied temp & precip
  dec_cstate(:,1) = (/ 2.0d0, 1.5d0, 1.0d0, 0.5d0, 5.0d0 /)
  dec_cstate(:,2) = (/ 2.0d0, 1.5d0, 1.0d0, 0.5d0, 5.0d0 /)
  dec_cstate(:,3) = (/ 2.0d0, 1.5d0, 1.0d0, 0.5d0, 5.0d0 /)
  dec_cstate(:,4) = (/ 2.0d0, 1.5d0, 1.0d0, 0.5d0, 5.0d0 /)
  dec_cstate(:,5) = (/ 2.0d0, 1.5d0, 1.0d0, 0.5d0, 5.0d0 /)
  ! Case 6: same temp as case 1 but low precip
  dec_cstate(:,6) = (/ 2.0d0, 1.5d0, 1.0d0, 0.5d0, 5.0d0 /)
  ! Case 7: same temp as case 1 but high precip
  dec_cstate(:,7) = (/ 2.0d0, 1.5d0, 1.0d0, 0.5d0, 5.0d0 /)
  ! Case 8: near-zero carbon → triggers totc < 1e-6 branch (ntend=0)
  dec_cstate(:,8) = (/ 1.0d-7, 1.0d-7, 1.0d-7, 1.0d-7, 1.0d-7 /)
  ! Case 9: unusual humus N:C branch only (cstate(5)*nc_h_max > nstate)
  dec_cstate(:,9) = (/ 2.0d0, 1.5d0, 1.0d0, 0.5d0, 5.0d0 /)
  ! Case 10: CUE lower floor only (nc_som low, humus small to avoid nc_h branch)
  dec_cstate(:,10) = (/ 4.0d0, 3.0d0, 2.0d0, 1.0d0, 1.0d-2 /)

  ! Nitrogen state
  dec_nstate(1:7) = 0.5d0
  dec_nstate(8) = 1.0d-8  ! near-zero
  dec_nstate(9) = 0.4d0   ! nc_h unusual, cue stays unclamped
  dec_nstate(10) = 5.0d-3 ! cue lower floor, nc_h branch stays false

  do i = 1, NDECOMP
    call decompose(dec_param, dec_timestep_days, dec_tempr_c(i), &
                   dec_precip_day(i), dec_cstate(:,i), dec_nstate(i), &
                   dec_ctend, dec_ntend)
    write(10, '(A)', advance='no') '{"fn":"decompose","inputs":{"param":'
    call write_vec_json(10, dec_param, 35)
    write(10, '(A,ES22.15,A,ES22.15,A,ES22.15,A)', advance='no') &
      ',"timestep_days":', dec_timestep_days, &
      ',"tempr_c":', dec_tempr_c(i), &
      ',"precip_day":', dec_precip_day(i), ',"cstate":'
    call write_vec_json(10, dec_cstate(:,i), 5)
    write(10, '(A,ES22.15)', advance='no') ',"nstate":', dec_nstate(i)
    write(10, '(A)', advance='no') '},"output":{"ctend":'
    call write_vec_json(10, dec_ctend, 5)
    write(10, '(A,ES22.15,A)') ',"ntend":', dec_ntend, '}}'
    nrec = nrec + 1
  end do

  ! ================================================================
  ! 27. alloc_hypothesis_2 — daily biomass allocation
  ! ================================================================
  ! Save pft_type since allocation tests modify this module variable
  ah2_pft_saved = pft_type

  do i = 1, N_AH2
    ! --- Common defaults (reset each iteration, since ah2 modifies in-place) ---
    ah2_ap%cratio_resp    = 0.4d0
    ah2_ap%cratio_leaf    = 0.8d0
    ah2_ap%cratio_root    = 0.2d0
    ah2_ap%cratio_biomass = 0.42d0
    ah2_ap%harvest_index  = 0.5d0
    ah2_ap%turnover_cleaf = 0.41d0/365.0d0
    ah2_ap%turnover_croot = 0.41d0/365.0d0
    ah2_ap%sla            = 10.0d0
    ah2_ap%q10            = 2.0d0
    ah2_ap%invert_option  = 0.0d0

    ah2_md%management_type    = 0
    ah2_md%management_c_input  = 0.0d0
    ah2_md%management_c_output = 0.0d0
    ah2_md%management_n_input  = 0.0d0
    ah2_md%management_n_output = 0.0d0

    ah2_temp  = 15.0d0
    ah2_gpp   = 3.0d-7
    ah2_rdark = 3.0d-8
    ah2_npp   = 0.0d0
    ah2_resp  = 0.0d0
    ah2_croot = 0.05d0
    ah2_cleaf = 0.10d0
    ah2_cstem = 0.02d0
    ah2_cgrain = 0.0d0
    ah2_litter_cleaf = 0.0d0
    ah2_litter_croot = 0.0d0
    ah2_compost = 0.0d0
    ah2_lai   = 0.10d0/0.42d0 * 10.0d0  ! = cleaf/cbiomass*sla
    ah2_above = 0.0d0
    ah2_below = 0.0d0
    ah2_yield = 0.0d0
    ah2_grain_fill = 0.0d0
    ah2_pheno = 1
    pft_type  = "grass"

    ! --- Per-case overrides ---
    select case(i)
    case(1)  ! Growth, grass, inv=0, no management
      ! defaults suffice
    case(2)  ! Growth, oat, inv=0, no management (grain logic)
      pft_type = "oat"
      ah2_cgrain = 0.005d0
      ah2_grain_fill = 2.0d-8
    case(3)  ! Harvest grass, inv=0
      ah2_md%management_type = 1
      ah2_md%management_c_output = 5.0d-8
    case(4)  ! Harvest oat, inv=0 (reset all pools)
      pft_type = "oat"
      ah2_md%management_type = 1
      ah2_md%management_c_output = 5.0d-8
      ah2_cgrain = 0.005d0
      ah2_grain_fill = 2.0d-8
    case(5)  ! Grazing, grass, inv=0
      ah2_md%management_type = 3
      ah2_md%management_c_output = 5.0d-8
      ah2_md%management_c_input  = 1.0d-8
    case(6)  ! Organic input, grass, inv=0
      ah2_md%management_type = 4
      ah2_md%management_c_input = 1.0d-8
    case(7)  ! Growth, grass, inv=1 (litter/cleaf not updated by alloc)
      ah2_ap%invert_option = 1.0d0
    case(8)  ! Growth, grass, inv=2 (litter/cleaf not updated by alloc)
      ah2_ap%invert_option = 2.0d0
    case(9)  ! Dormancy (pheno=2, all biomass -> litter)
      ah2_pheno = 2
    case(10)  ! Cold temperature (Q10 effect), grass, inv=0
      ah2_temp = -5.0d0
    case(11)  ! Grazing, oat, inv=0 (grazing has no pft check for cleaf)
      pft_type = "oat"
      ah2_md%management_type = 3
      ah2_md%management_c_output = 5.0d-8
      ah2_md%management_c_input  = 1.0d-8
      ah2_cgrain = 0.005d0
      ah2_grain_fill = 2.0d-8
    case(12)  ! Harvest grass, inv=1 (test inv guard skips cleaf deduction)
      ah2_ap%invert_option = 1.0d0
      ah2_md%management_type = 1
      ah2_md%management_c_output = 5.0d-8
    case(13)  ! Grazing grass, inv=1 (test inv guard skips cleaf deduction)
      ah2_ap%invert_option = 1.0d0
      ah2_md%management_type = 3
      ah2_md%management_c_output = 5.0d-8
      ah2_md%management_c_input  = 1.0d-8
    end select

    ! Save input state snapshot
    ah2_sin = (/ ah2_npp, ah2_resp, ah2_croot, ah2_cleaf, ah2_cstem, &
                 ah2_cgrain, ah2_litter_cleaf, ah2_litter_croot, ah2_compost, &
                 ah2_lai, ah2_above, ah2_below, ah2_yield, ah2_grain_fill /)
    ah2_pheno_in = ah2_pheno  ! save before call (subroutine may modify)

    ! Call subroutine (modifies state in-place)
    call alloc_hypothesis_2(ah2_temp, ah2_gpp, ah2_npp, ah2_rdark, ah2_resp, &
                            ah2_croot, ah2_cleaf, ah2_cstem, ah2_cgrain, &
                            ah2_litter_cleaf, ah2_litter_croot, ah2_compost, &
                            ah2_above, ah2_below, ah2_yield, &
                            ah2_lai, ah2_ap, ah2_grain_fill, ah2_md, ah2_pheno)

    ! Save output state
    ah2_sout = (/ ah2_npp, ah2_resp, ah2_croot, ah2_cleaf, ah2_cstem, &
                  ah2_cgrain, ah2_litter_cleaf, ah2_litter_croot, ah2_compost, &
                  ah2_lai, ah2_above, ah2_below, ah2_yield, ah2_grain_fill /)

    ! Write JSONL record
    call log_ah2(10, ah2_temp, ah2_gpp, ah2_rdark, ah2_ap, ah2_md, &
                 ah2_sin, ah2_pheno_in, ah2_sout, ah2_pheno)
    nrec = nrec + 1
  end do

  ! ================================================================
  ! 28. invert_alloc — LAI-to-leaf-carbon inversion
  ! ================================================================
  do i = 1, N_INVALLOC
    ! --- Common defaults (reset each iteration) ---
    inv_ap%cratio_resp    = 0.4d0
    inv_ap%cratio_leaf    = 0.8d0
    inv_ap%cratio_root    = 0.2d0
    inv_ap%cratio_biomass = 0.42d0
    inv_ap%harvest_index  = 0.5d0
    inv_ap%turnover_cleaf = 0.41d0/365.0d0
    inv_ap%turnover_croot = 0.41d0/365.0d0
    inv_ap%sla            = 10.0d0
    inv_ap%q10            = 2.0d0
    inv_ap%invert_option  = 1.0d0

    inv_md%management_type    = 0
    inv_md%management_c_input  = 0.0d0
    inv_md%management_c_output = 0.0d0
    inv_md%management_n_input  = 0.0d0
    inv_md%management_n_output = 0.0d0

    inv_delta_lai    = 0.05d0       ! positive LAI change
    inv_rdark        = 3.0d-8
    inv_temp         = 15.0d0
    inv_gpp          = 3.0d-7
    inv_litter_cleaf = 0.0d0
    inv_cleaf        = 0.10d0
    inv_cstem        = 0.02d0
    inv_pheno        = 1
    pft_type         = "grass"

    ! --- Per-case overrides ---
    select case(i)
    case(1)  ! Option 1, no management, grass
      ! defaults suffice
    case(2)  ! Option 1, harvest grass
      inv_md%management_type = 1
      inv_md%management_c_output = 5.0d-8
    case(3)  ! Option 1, harvest oat (non-grass)
      pft_type = "oat"
      inv_md%management_type = 1
      inv_md%management_c_output = 5.0d-8
    case(4)  ! Option 1, grazing grass
      inv_md%management_type = 3
      inv_md%management_c_output = 5.0d-8
      inv_md%management_c_input  = 1.0d-8
    case(5)  ! Option 1, grazing oat (non-grass)
      pft_type = "oat"
      inv_md%management_type = 3
      inv_md%management_c_output = 5.0d-8
      inv_md%management_c_input  = 1.0d-8
    case(6)  ! Option 1, gpp below threshold (no cratio update)
      inv_gpp = 1.0d-9
    case(7)  ! Option 2, no management, grass
      inv_ap%invert_option = 2.0d0
    case(8)  ! Option 2, harvest grass
      inv_ap%invert_option = 2.0d0
      inv_md%management_type = 1
      inv_md%management_c_output = 5.0d-8
    case(9)  ! Option 2, harvest oat (non-grass)
      inv_ap%invert_option = 2.0d0
      pft_type = "oat"
      inv_md%management_type = 1
      inv_md%management_c_output = 5.0d-8
    case(10)  ! Option 2, grazing grass
      inv_ap%invert_option = 2.0d0
      inv_md%management_type = 3
      inv_md%management_c_output = 5.0d-8
      inv_md%management_c_input  = 1.0d-8
    case(11)  ! Option 2, grazing oat (non-grass)
      inv_ap%invert_option = 2.0d0
      pft_type = "oat"
      inv_md%management_type = 3
      inv_md%management_c_output = 5.0d-8
      inv_md%management_c_input  = 1.0d-8
    case(12)  ! Option 2, cleaf near zero (below threshold)
      inv_ap%invert_option = 2.0d0
      inv_cleaf = 1.0d-6
    case(13)  ! Pheno=2 (inactive, no action)
      inv_pheno = 2
    case(14)  ! Option 1, organic fertilizer (falls to else -> no mgmt path)
      inv_md%management_type = 4
      inv_md%management_c_input = 1.0d-8
    end select

    ! Save input snapshot:
    ! [delta_lai, litter_cleaf, cleaf, cratio_leaf, cratio_root]
    inv_sin = (/ inv_delta_lai, inv_litter_cleaf, inv_cleaf, &
                 inv_ap%cratio_leaf, inv_ap%cratio_root /)

    ! Call subroutine (modifies in-place)
    call invert_alloc(inv_delta_lai, inv_ap, inv_rdark, inv_temp, &
                      inv_litter_cleaf, inv_gpp, inv_cleaf, &
                      inv_cstem, inv_md, inv_pheno)

    ! Save output snapshot:
    ! [delta_lai, litter_cleaf, cleaf, cratio_leaf, cratio_root, turnover_cleaf, cstem, pheno]
    inv_sout = (/ inv_delta_lai, inv_litter_cleaf, inv_cleaf, &
                  inv_ap%cratio_leaf, inv_ap%cratio_root, &
                  inv_ap%turnover_cleaf, inv_cstem, dble(inv_pheno) /)

    ! Write JSONL record
    call log_invalloc(10, inv_delta_lai, inv_rdark, inv_temp, inv_gpp, &
                      inv_ap, inv_md, inv_sin, inv_pheno, inv_sout)
    nrec = nrec + 1
  end do

  ! Restore pft_type
  pft_type = ah2_pft_saved

  ! --- Close JSONL file and report summary to stdout ---
  close(10)
  write(*, '(A,I0,A)') 'Emitted ', nrec, ' records to fixtures.jsonl'

CONTAINS

  ! ================================================================
  ! Build the mod5c20 coefficient matrix and multiply by time.
  ! This duplicates only the matrix-construction portion of yasso20.mod5c20
  ! so matrixexp fixtures can come from a real yearly Yasso execution path.
  ! ================================================================
  SUBROUTINE build_mod5c20_matrix(theta, time, temp, prec, d, leac, At)
    real(8), dimension(35), intent(in) :: theta
    real(8), intent(in) :: time, prec, d, leac
    real(8), dimension(12), intent(in) :: temp
    real(8), dimension(5,5), intent(out) :: At
    real(8), dimension(5,5) :: A
    real(8), parameter :: tol = 1.0d-12
    real(8) :: tem, temN, temH, size_dep
    integer :: i

    A = 0.0d0
    tem = 0.0d0
    temN = 0.0d0
    temH = 0.0d0

    do i = 1,12
      tem = tem + exp(theta(22) * temp(i) + theta(23) * temp(i) ** 2.0d0)
      temN = temN + exp(theta(24) * temp(i) + theta(25) * temp(i) ** 2.0d0)
      temH = temH + exp(theta(26) * temp(i) + theta(27) * temp(i) ** 2.0d0)
    end do

    tem = tem * (1.0d0 - exp(theta(28) * prec / 1000.0d0)) / 12.0d0
    temN = temN * (1.0d0 - exp(theta(29) * prec / 1000.0d0)) / 12.0d0
    temH = temH * (1.0d0 - exp(theta(30) * prec / 1000.0d0)) / 12.0d0
    size_dep = min(1.0d0, &
      (1.0d0 + theta(33) * d + theta(34) * d ** 2.0d0) ** (-abs(theta(35))))

    if (tem <= tol) then
      At = 0.0d0
      return
    end if

    do i = 1,3
      A(i,i) = -abs(theta(i)) * tem * size_dep
    end do
    A(4,4) = -abs(theta(4)) * temN * size_dep
    A(1,2) = theta(5) * abs(A(2,2))
    A(1,3) = theta(6) * abs(A(3,3))
    A(1,4) = theta(7) * abs(A(4,4))
    A(2,1) = theta(8) * abs(A(1,1))
    A(2,3) = theta(9) * abs(A(3,3))
    A(2,4) = theta(10) * abs(A(4,4))
    A(3,1) = theta(11) * abs(A(1,1))
    A(3,2) = theta(12) * abs(A(2,2))
    A(3,4) = theta(13) * abs(A(4,4))
    A(4,1) = theta(14) * abs(A(1,1))
    A(4,2) = theta(15) * abs(A(2,2))
    A(4,3) = theta(16) * abs(A(3,3))
    A(5,5) = -abs(theta(32)) * temH
    do i = 1,4
      A(5,i) = theta(31) * abs(A(i,i))
    end do

    do i = 1,4
      A(i,i) = A(i,i) + leac * prec / 1000.0d0
    end do

    At = A * time
  END SUBROUTINE build_mod5c20_matrix

  ! ================================================================
  ! Helper: write a 5×5 matrix as nested JSON array [[r1],[r2],...]
  ! ================================================================
  SUBROUTINE write_mat5_json(u, mat)
    integer, intent(in) :: u
    real(8), dimension(5,5), intent(in) :: mat
    integer :: r, c
    write(u, '(A)', advance='no') '['
    do r = 1, 5
      write(u, '(A)', advance='no') '['
      do c = 1, 5
        write(u, '(ES22.15)', advance='no') mat(r,c)
        if (c < 5) write(u, '(A)', advance='no') ','
      end do
      write(u, '(A)', advance='no') ']'
      if (r < 5) write(u, '(A)', advance='no') ','
    end do
    write(u, '(A)', advance='no') ']'
  END SUBROUTINE write_mat5_json

  ! ================================================================
  ! Helper: write an n-element vector as JSON array [v1,v2,...]
  ! ================================================================
  SUBROUTINE write_vec_json(u, vec, n)
    integer, intent(in) :: u, n
    real(8), dimension(n), intent(in) :: vec
    integer :: vi
    write(u, '(A)', advance='no') '['
    do vi = 1, n
      write(u, '(ES22.15)', advance='no') vec(vi)
      if (vi < n) write(u, '(A)', advance='no') ','
    end do
    write(u, '(A)', advance='no') ']'
  END SUBROUTINE write_vec_json

  ! ================================================================
  ! Helper: write alloc_hypothesis_2 JSONL record
  ! state(14) = [npp, resp, croot, cleaf, cstem, cgrain,
  !              litter_cleaf, litter_croot, compost,
  !              lai, above, below, yield, grain_fill]
  ! ================================================================
  SUBROUTINE log_ah2(u, temp, gpp, rdark, ap, md, st_in, pheno_in, st_out, pheno_out)
    integer, intent(in) :: u, pheno_in, pheno_out
    real(8), intent(in) :: temp, gpp, rdark
    type(alloc_para_type), intent(in) :: ap
    type(management_data_type), intent(in) :: md
    real(8), dimension(14), intent(in) :: st_in, st_out
    ! --- inputs ---
    write(u, '(A)', advance='no') '{"fn":"alloc_hypothesis_2","inputs":{'
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      '"temp_day":', temp, ',"gpp_day":', gpp, ',"leaf_rdark_day":', rdark
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"npp_day":', st_in(1), ',"auto_resp":', st_in(2), ',"croot":', st_in(3)
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"cleaf":', st_in(4), ',"cstem":', st_in(5), ',"cgrain":', st_in(6)
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"litter_cleaf":', st_in(7), ',"litter_croot":', st_in(8), ',"compost":', st_in(9)
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"lai":', st_in(10), ',"abovebiomass":', st_in(11), ',"belowbiomass":', st_in(12)
    write(u, '(A,ES22.15,A,ES22.15)', advance='no') &
      ',"yield":', st_in(13), ',"grain_fill":', st_in(14)
    ! alloc_para fields
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"cratio_resp":', ap%cratio_resp, ',"cratio_leaf":', ap%cratio_leaf, &
      ',"cratio_root":', ap%cratio_root
    write(u, '(A,ES22.15,A,ES22.15)', advance='no') &
      ',"cratio_biomass":', ap%cratio_biomass, ',"harvest_index":', ap%harvest_index
    write(u, '(A,ES22.15,A,ES22.15)', advance='no') &
      ',"turnover_cleaf":', ap%turnover_cleaf, ',"turnover_croot":', ap%turnover_croot
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"sla":', ap%sla, ',"q10":', ap%q10, ',"invert_option":', ap%invert_option
    ! management fields
    write(u, '(A,I0,A,ES22.15,A,ES22.15)', advance='no') &
      ',"management_type":', md%management_type, &
      ',"management_c_input":', md%management_c_input, &
      ',"management_c_output":', md%management_c_output
    ! pft_type and pheno_stage
    write(u, '(A,A,A,A,I0)', advance='no') &
      ',"pft_type":"', trim(pft_type), '"', ',"pheno_stage":', pheno_in
    ! --- output ---
    write(u, '(A)', advance='no') '},"output":{'
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      '"npp_day":', st_out(1), ',"auto_resp":', st_out(2), ',"croot":', st_out(3)
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"cleaf":', st_out(4), ',"cstem":', st_out(5), ',"cgrain":', st_out(6)
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"litter_cleaf":', st_out(7), ',"litter_croot":', st_out(8), ',"compost":', st_out(9)
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"lai":', st_out(10), ',"abovebiomass":', st_out(11), ',"belowbiomass":', st_out(12)
    write(u, '(A,ES22.15,A,ES22.15,A,I0,A)') &
      ',"yield":', st_out(13), ',"grain_fill":', st_out(14), ',"pheno_stage":', pheno_out, '}}'
  END SUBROUTINE log_ah2

  ! ================================================================
  ! Helper: write invert_alloc JSONL record
  ! sin(5) = [delta_lai, litter_cleaf, cleaf, cratio_leaf, cratio_root]
  ! sout(8) = [delta_lai, litter_cleaf, cleaf, cratio_leaf, cratio_root,
  !            turnover_cleaf, cstem, pheno]
  ! ================================================================
  SUBROUTINE log_invalloc(u, delta_lai, rdark, temp, gpp, ap, md, st_in, pheno_in, st_out)
    integer, intent(in) :: u, pheno_in
    real(8), intent(in) :: delta_lai, rdark, temp, gpp
    type(alloc_para_type), intent(in) :: ap
    type(management_data_type), intent(in) :: md
    real(8), dimension(5), intent(in) :: st_in
    real(8), dimension(8), intent(in) :: st_out
    ! --- inputs ---
    write(u, '(A)', advance='no') '{"fn":"invert_alloc","inputs":{'
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      '"delta_lai":', delta_lai, ',"leaf_rdark_day":', rdark, ',"temp_day":', temp
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"gpp_day":', gpp, ',"litter_cleaf":', st_in(2), ',"cleaf":', st_in(3)
    write(u, '(A,ES22.15)', advance='no') ',"cstem":', st_out(7)
    ! alloc_para fields
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"cratio_resp":', ap%cratio_resp, ',"cratio_leaf":', st_in(4), &
      ',"cratio_root":', st_in(5)
    write(u, '(A,ES22.15,A,ES22.15)', advance='no') &
      ',"cratio_biomass":', ap%cratio_biomass, ',"harvest_index":', ap%harvest_index
    write(u, '(A,ES22.15,A,ES22.15)', advance='no') &
      ',"turnover_cleaf":', ap%turnover_cleaf, ',"turnover_croot":', ap%turnover_croot
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"sla":', ap%sla, ',"q10":', ap%q10, ',"invert_option":', ap%invert_option
    ! management fields
    write(u, '(A,I0,A,ES22.15,A,ES22.15)', advance='no') &
      ',"management_type":', md%management_type, &
      ',"management_c_input":', md%management_c_input, &
      ',"management_c_output":', md%management_c_output
    ! pft_type and pheno_stage
    write(u, '(A,A,A,A,I0)', advance='no') &
      ',"pft_type":"', trim(pft_type), '"', ',"pheno_stage":', pheno_in
    ! --- output ---
    write(u, '(A)', advance='no') '},"output":{'
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      '"delta_lai":', st_out(1), ',"litter_cleaf":', st_out(2), ',"cleaf":', st_out(3)
    write(u, '(A,ES22.15,A,ES22.15,A,ES22.15)', advance='no') &
      ',"cratio_leaf":', st_out(4), ',"cratio_root":', st_out(5), &
      ',"turnover_cleaf":', st_out(6)
    write(u, '(A)') '}}'
  END SUBROUTINE log_invalloc

  ! ================================================================
  ! Duplicated private functions from water_mod.f90
  ! (exact copies to produce identical reference values)
  ! ================================================================

  SUBROUTINE harness_e_sat(T, P, s, g, esat)
    real(8), intent(in) :: T, P
    real(8), intent(out) :: esat, s, g

    real(8) :: NT, cp, Lambda

    NT = 273.15d0
    cp = 1004.67d0

    Lambda = 1.0d3 * (3147.5d0 - 2.37d0 * (T + NT))
    esat = 1.0d3 * (0.6112d0 * exp((17.67d0 * T) / (T + 273.16d0 - 29.66d0)))
    s = 17.502d0 * 240.97d0 * esat / ((240.97d0 + T) ** 2)
    g = P * cp / (0.622d0 * Lambda)
  END SUBROUTINE harness_e_sat

  real(8) FUNCTION harness_penman_monteith(AE, D, T, Gs, Ga, P)
    real(8), intent(in) :: AE, D, T, Gs, Ga, P

    real(8) :: cp, rho, s, g, esat

    cp  = 1004.67d0
    rho = 1.25d0

    call harness_e_sat(T, P, s, g, esat)

    harness_penman_monteith = (s * AE + rho * cp * Ga * D) / &
                              (s + g * (1.0d0 + Ga / Gs))
    ! PORT-BRANCH: water.penman_monteith.le_floor (harness duplicate)
    ! Condition: result < 0 -> clamp latent heat flux to 0 (harness duplicate)
    harness_penman_monteith = max(harness_penman_monteith, 0.0d0)
  END FUNCTION harness_penman_monteith

  ! ================================================================
  ! Duplicated from water_mod.f90 (aerodynamics is PRIVATE)
  ! Exact copy of the water_mod version (with LAI > eps guard)
  ! ================================================================
  SUBROUTINE harness_aerodynamics(LAI, Uo, ra, rb, ras, ustar, Uh, Ug, spafhy_para)
    real(8), intent(in)  :: LAI, Uo
    type(spafhy_para_type), intent(in) :: spafhy_para
    real(8), intent(out) :: ra, rb, ras, ustar, Uh, Ug

    real(8) :: zm1, zg1, alpha1, d, zom, zov, zosv, zn
    real(8) :: kv = 0.4d0
    real(8) :: beta_aero = 285.0d0
    real(8) :: eps = 1e-16

    zm1 = spafhy_para%hc + spafhy_para%zmeas
    ! PORT-BRANCH: water.aerodynamics.ground_height_cap (harness duplicate)
    ! Condition: zground > 0.1*hc -> cap ground height at 10% of canopy (harness duplicate)
    zg1 = min(spafhy_para%zground, 0.1d0 * spafhy_para%hc)
    alpha1 = LAI / 2.0d0
    d = 0.66d0 * spafhy_para%hc
    zom = 0.123d0 * spafhy_para%hc
    zov = 0.1d0 * zom
    zosv = 0.1d0 * spafhy_para%zo_ground

    ustar = Uo * kv / log((zm1 - d) / zom)
    Uh = ustar / kv * log((spafhy_para%hc - d) / zom)

    ! PORT-BRANCH: water.aerodynamics.zn_cap (harness duplicate)
    ! Condition: zg1/hc > 1.0 -> cap normalized ground height at 1.0 (harness duplicate)
    zn = min(zg1 / spafhy_para%hc, 1.0d0)
    Ug = Uh * exp(alpha1 * (zn - 1.0d0))

    ra = 1.0d0 / (kv**2.0d0 * Uo) * log((zm1 - d) / zom) * log((zm1 - d) / zov)

    ! PORT-BRANCH: water.aerodynamics.rb_lai_guard (harness duplicate)
    ! Condition: LAI <= eps -> boundary-layer resistance forced to 0 (harness duplicate)
    if (LAI > eps) then
      rb = 1.0d0/LAI * beta_aero * ((spafhy_para%w_leaf / Uh) * &
           (alpha1 / (1.0d0 - exp(-alpha1 / 2.0d0))))**0.5d0
    else
      rb = 0.0d0
    end if

    ras = 1.0d0 / (kv**2.0d0 * Ug) * log(spafhy_para%zground / spafhy_para%zo_ground) * &
          log(spafhy_para%zground / zosv)

    ra = ra + rb
  END SUBROUTINE harness_aerodynamics

  ! ================================================================
  ! Duplicated from water_mod.f90 (ground_evaporation is PRIVATE)
  ! Exact copy with penman_monteith -> harness_penman_monteith
  ! ================================================================
  SUBROUTINE harness_ground_evaporation(canopywater_state, canopywater_flux, &
      soilwater_state, spafhy_para, T, AE, VPD, Ras, P)
    real(8), intent(in) :: T, AE, VPD, Ras, P
    type(canopywater_flux_type), intent(inout) :: canopywater_flux
    type(soilwater_state_type), intent(in) :: soilwater_state
    type(canopywater_state_type), intent(in) :: canopywater_state
    type(spafhy_para_type), intent(in) :: spafhy_para

    real(8) :: Lv, erate, Gas, eps
    eps = 1.0d-16
    Lv = 1.0d3 * (3147.5d0 - 2.37d0 * (T + 273.15d0))
    Gas = 1.0d0 / Ras

    erate = (time_step * 3600.0d0) * soilwater_state%beta * &
            harness_penman_monteith(AE, VPD, T, spafhy_para%gsoil, Gas, P) / Lv

    ! maximum equals available water
    canopywater_flux%SoilEvap = min(soilwater_state%WatSto, erate)

    ! PORT-BRANCH: water.ground_evaporation.snow_floor_zero (harness duplicate)
    ! Condition: SWE > eps -> no evaporation from floor if snow on ground (harness duplicate)
    if (canopywater_state%SWE > eps) then
      canopywater_flux%SoilEvap = 0.0d0
    end if
  END SUBROUTINE harness_ground_evaporation

  ! ================================================================
  ! Duplicated from water_mod.f90 (canopy_water_snow is PRIVATE)
  ! Exact copy with penman_monteith -> harness_penman_monteith
  ! ================================================================
  SUBROUTINE harness_canopy_water_snow(canopywater_state, canopywater_flux, &
      spafhy_para, T, Pre, AE, D, Ra, U, LAI, P)
    real(8), intent(in) :: T, Pre, AE, D, Ra, U, LAI, P
    type(canopywater_state_type), intent(inout) :: canopywater_state
    type(canopywater_flux_type), intent(inout) :: canopywater_flux
    type(spafhy_para_type), intent(in) :: spafhy_para

    real(8) :: fW, fS, Tmin, Tmax, Tmelt, wmax_tot, wmaxsnow_tot
    real(8) :: Ga, Ce, Sh, gi, erate, gs, Sice, Sliq, swe_i, swe_l
    real(8) :: Lv, Ls, SWEo, Wo, Prec, W
    real(8) :: eps
    real(8) :: Unload, Interc, Melt, Freeze, Trfall
    real(8) :: CanopyEvap, PotInfil

    Sice = 0.0d0
    Sliq = 0.0d0
    Unload = 0.0d0
    Interc = 0.0d0
    Melt = 0.0d0
    Freeze = 0.0d0
    Trfall = 0.0d0
    CanopyEvap = 0.0d0
    PotInfil = 0.0d0
    eps = 1.0d-16

    Tmin = 0.0d0
    Tmax = 1.0d0
    Tmelt = 0.0d0

    ! PORT-BRANCH: water.canopy_water_snow.precip_phase (harness duplicate)
    ! Condition: T<=Tmin -> all snow; T>=Tmax -> all rain; between -> mixed (harness duplicate)
    if (T <= Tmin) then
      fS = 1.0d0
      fW = 0.0d0
    else if (T >= Tmax) then
      fW = 1.0d0
      fS = 0.0d0
    else if ((T > Tmin) .and. (T < Tmax)) then
      fW = (T - Tmin) / (Tmax - Tmin)
      fS = 1.0d0 - fW
    end if

    wmax_tot = spafhy_para%wmax * LAI
    wmaxsnow_tot = spafhy_para%wmaxsnow * LAI

    Lv = 1.0d3 * (3147.5d0 - 2.37d0 * (T + 273.15d0))
    Ls = Lv + 3.3d5

    Prec = Pre * time_step * 3600.0d0

    Wo = canopywater_state%CanopyStorage
    SWEo = canopywater_state%SWE

    W = Wo
    swe_i = canopywater_state%swe_i
    swe_l = canopywater_state%swe_l

    Ga = 1.0d0 / Ra

    ! PORT-BRANCH: water.canopy_water_snow.lai_evap_guard (harness duplicate)
    ! Condition: LAI <= eps -> no canopy evaporation/sublimation (harness duplicate)
    if (LAI > eps) then
      Ce = 0.01d0 * ((W + eps) / wmaxsnow_tot)**(-0.4d0)
      Sh = (1.79d0 + 3.0d0 * U**0.5d0)
      gi = Sh * W * Ce / 7.68d0 + eps

      erate = 0.0d0
      ! PORT-BRANCH: water.canopy_water_snow.sublim_vs_evap (harness duplicate)
      ! Condition: Prec==0 & T<=Tmin -> sublimation; Prec==0 & T>Tmin -> evaporation (harness duplicate)
      if ((Prec == 0.0d0) .and. (T <= Tmin)) then
        erate = (time_step * 3600.0d0) / Ls * &
                harness_penman_monteith(AE, D, T, gi, Ga, P)
      else if ((Prec == 0.0d0) .and. (T > Tmin)) then
        gs = 1.0d6
        erate = (time_step * 3600.0d0) / Lv * &
                harness_penman_monteith(AE, D, T, gs, Ga, P)
      end if
    else
      erate = 0.0d0
    end if

    ! PORT-BRANCH: water.canopy_water_snow.snow_unloading (harness duplicate)
    ! Condition: T >= Tmin -> unload excess beyond wmax_tot (harness duplicate)
    if (T >= Tmin) then
      Unload = max(W - wmax_tot, 0.0d0)
      W = W - Unload
    end if

    ! PORT-BRANCH: water.canopy_water_snow.interception_phase (harness duplicate)
    ! Condition: T < Tmin -> snow interception capacity; else -> liquid capacity (harness duplicate)
    if (T < Tmin) then
      if (LAI > eps) then
        Interc = (wmaxsnow_tot - W) * (1.0d0 - exp(-Prec / wmaxsnow_tot))
      end if
    else if (T >= Tmin) then
      if (LAI > eps) then
        Interc = max(0.0d0, (wmax_tot - W)) * (1.0d0 - exp(-Prec / wmax_tot))
      end if
    end if

    W = W + Interc
    CanopyEvap = min(erate, W + eps)
    W = W - CanopyEvap
    Trfall = Prec + Unload - Interc

    ! PORT-BRANCH: water.canopy_water_snow.melt_freeze (harness duplicate)
    ! Condition: T>=Tmelt -> melt ice; T<Tmelt & swe_l>0 -> freeze liquid; else -> no phase change (harness duplicate)
    if (T >= Tmelt) then
      Melt = min(swe_i, spafhy_para%kmelt * (time_step * 3600.0d0) * (T - Tmelt))
      Freeze = 0.0d0
    else if (T < Tmelt .and. swe_l > 0.0d0) then
      Freeze = min(swe_l, spafhy_para%kfreeze * (time_step * 3600.0d0) * (Tmelt - T))
      Melt = 0.0d0
    else
      Freeze = 0.0d0
      Melt = 0.0d0
    end if

    Sice = max(0.0d0, swe_i + fS * Trfall + Freeze - Melt)
    Sliq = max(0.0d0, swe_l + fW * Trfall - Freeze + Melt)

    PotInfil = max(0.0d0, Sliq - Sice * spafhy_para%frac_snowliq)
    Sliq = max(0.0d0, Sliq - PotInfil)

    canopywater_state%CanopyStorage = W
    canopywater_state%swe_l = Sliq
    canopywater_state%swe_i = Sice
    canopywater_state%SWE = canopywater_state%swe_l + canopywater_state%swe_i

    canopywater_flux%Unloading = Unload
    canopywater_flux%Interception = Interc
    canopywater_flux%CanopyEvap = CanopyEvap
    canopywater_flux%Throughfall = Trfall
    canopywater_flux%PotInfiltration = PotInfil
    canopywater_flux%Melt = Melt
    canopywater_flux%Freeze = Freeze

    canopywater_flux%mbe = (canopywater_state%CanopyStorage + canopywater_state%SWE) - &
                           (Wo + SWEo) - (Prec - canopywater_flux%CanopyEvap - &
                           canopywater_flux%PotInfiltration)
  END SUBROUTINE harness_canopy_water_snow

  ! ================================================================
  ! Duplicated from wrapper_yasso.f90 (to avoid heavy yasso20 deps)
  ! Note: -freal-4-real-8 promotes the real*4 alpha constants to real*8
  ! Canonical constant values: packages/svmc-ref/constants/water.json
  ! ================================================================
  SUBROUTINE exponential_smooth_met(met_daily, met_rolling, met_ind)
    real(8), intent(in)    :: met_daily(:)
    real(8), intent(inout) :: met_rolling(:)
    integer, intent(inout) :: met_ind
    real(8) :: alpha_smooth1, alpha_smooth2
    alpha_smooth1 = 0.01d0
    alpha_smooth2 = 0.0016d0

    ! PORT-BRANCH: yasso.exponential_smooth_met.init_vs_smooth (harness duplicate)
    ! Condition: met_ind == 1 -> initialize with daily value; else smooth rolling state (harness duplicate)
    if (met_ind == 1) then
       met_rolling(:) = met_daily(:)
       met_ind = met_ind + 1
    else
       met_rolling(1) = alpha_smooth1 * met_daily(1) + (1.0d0 - alpha_smooth1) * met_rolling(1)
       met_rolling(2) = alpha_smooth2 * met_daily(2) + (1.0d0 - alpha_smooth2) * met_rolling(2)
    end if
  END SUBROUTINE exponential_smooth_met

END PROGRAM harness
