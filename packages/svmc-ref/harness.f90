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
                       quadratic
  use water_mod, only: soil_water_retention_curve, soil_hydraulic_conductivity
  use yasso, only: inputs_to_fractions, statesize_yasso
  use readvegpara_mod, only: par_plant_type, par_cost_type, par_env_type, &
                             par_photosynth_type, optimizer_type, kphio, &
                             opt_hypothesis
  use readsoilpara_mod, only: spafhy_para_type
  use readctrl_mod, only: time_step

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

  ! --- Close JSONL file and report summary to stdout ---
  close(10)
  write(*, '(A,I0,A)') 'Emitted ', nrec, ' records to fixtures.jsonl'

CONTAINS

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
