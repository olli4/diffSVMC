! Stub for readvegpara_mod — provides types, constants, and variables
! without namelist I/O.
!
! Provenance: derived from vendor/SVMC/src/readvegpara_mod.f90
! The svmc_wrapper sets all variables directly before the integration loop.

MODULE readvegpara_mod

  implicit none

  ! Derived types (used by phydro_mod)
  type, public :: par_plant_type
    real(8) :: conductivity
    real(8) :: psi50
    real(8) :: b
  end type par_plant_type

  type, public :: par_cost_type
    real(8) :: alpha
    real(8) :: gamma
  end type par_cost_type

  type, public :: par_env_type
    real(8) :: viscosity_water
    real(8) :: density_water
    real(8) :: patm
    real(8) :: tc
    real(8) :: vpd
  end type par_env_type

  type, public :: par_photosynth_type
    real(8) :: kmm
    real(8) :: gammastar
    real(8) :: phi0
    real(8) :: Iabs
    real(8) :: ca
    real(8) :: patm
    real(8) :: delta
  end type par_photosynth_type

  type, public :: optimizer_type
    real(8) :: logjmax
    real(8) :: dpsi
  end type optimizer_type

  ! Constants
  real(8) :: kphio = 0.087182
  real(8) :: k = 0.5
  real(8) :: c_molmass = 12.0107
  real(8) :: h2o_molmass = 18.01528

  ! Module variables (set by wrapper)
  character(len=256), public :: opt_hypothesis = 'PM'
  character(len=256), public :: pft_type = ''
  integer  :: num_pft = 1
  real(8)  :: conductivity, psi50, b, alpha, gamma, rdark

contains

  subroutine readvegpara_namelist
    ! No-op stub — variables are set directly by the wrapper
  end subroutine readvegpara_namelist

END MODULE readvegpara_mod
