! Stub for readsoilpara_mod — provides type definitions without namelist I/O.
!
! Provenance: derived from vendor/SVMC/src/readsoilpara_mod.f90
! The svmc_wrapper populates spafhy_para directly from R arguments.

MODULE readsoilpara_mod

  implicit none

  type, public :: soilwater_state_type
    real(8) :: WatSto
    real(8) :: MaxWatSto
    real(8) :: PondSto
    real(8) :: MaxPondSto
    real(8) :: FcSto
    real(8) :: Wliq
    real(8) :: Psi
    real(8) :: Sat
    real(8) :: Kh
    real(8) :: beta
  end type soilwater_state_type

  type, public :: soilwater_flux_type
    real(8) :: Infiltration
    real(8) :: Runoff
    real(8) :: Drainage
    real(8) :: LateralFlow
    real(8) :: ET
    real(8) :: mbe
  end type soilwater_flux_type

  type, public :: canopywater_state_type
    real(8) :: CanopyStorage
    real(8) :: SWE
    real(8) :: swe_i
    real(8) :: swe_l
  end type canopywater_state_type

  type, public :: canopywater_flux_type
    real(8) :: Throughfall
    real(8) :: Interception
    real(8) :: CanopyEvap
    real(8) :: Unloading
    real(8) :: SoilEvap
    real(8) :: ET
    real(8) :: Transpiration
    real(8) :: PotInfiltration
    real(8) :: Melt
    real(8) :: Freeze
    real(8) :: mbe
  end type canopywater_flux_type

  type, public :: spafhy_para_type
    real(8) :: maxpond
    real(8) :: soil_depth
    real(8) :: max_poros
    real(8) :: fc
    real(8) :: wp
    real(8) :: n_van
    real(8) :: watres
    real(8) :: alpha_van
    real(8) :: watsat
    real(8) :: ksat
    real(8) :: wmax
    real(8) :: wmaxsnow
    real(8) :: hc
    real(8) :: cf
    real(8) :: w_leaf
    real(8) :: rw
    real(8) :: rwmin
    real(8) :: gsoil
    real(8) :: kmelt
    real(8) :: kfreeze
    real(8) :: frac_snowliq
    real(8) :: zmeas
    real(8) :: zground
    real(8) :: zo_ground
  end type spafhy_para_type

  ! Module-level variables (used by readsoilhydro_namelist for compat)
  real(8) :: soil_depth, max_poros, fc, wp, ksat
  real(8) :: n_van, watres, alpha_van, watsat
  real(8) :: org_depth, org_poros, org_fc, org_sat
  real(8) :: maxpond
  real(8) :: wmax, wmaxsnow, hc, cf, w_leaf
  real(8) :: rw, rwmin, gsoil
  real(8) :: kmelt, kfreeze, frac_snowliq
  real(8) :: zmeas, zground, zo_ground

contains

  subroutine readsoilhydro_namelist(spafhy_para)
    type(spafhy_para_type), intent(inout) :: spafhy_para
    ! No-op stub — spafhy_para is populated by the wrapper
  end subroutine readsoilhydro_namelist

END MODULE readsoilpara_mod
