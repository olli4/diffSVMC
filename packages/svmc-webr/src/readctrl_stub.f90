! Stub for readctrl_mod — provides module variables without namelist I/O.
!
! Provenance: derived from vendor/SVMC/src/readctrl_mod.f90
! The svmc_wrapper sets all variables directly before the integration loop.

MODULE readctrl_mod

  implicit none

  public :: readctrl_namelist

  integer, parameter :: dp = selected_real_kind(P=15)

  integer :: start_date_day, start_date_hour, &
             end_date_day, end_date_hour
  integer :: num_sites
  real(8), dimension(1) :: lat_sites, lon_sites
  real    :: time_step, time_step_output
  character(len=256) :: output_dir, input_dir
  character(len=256) :: sites_name, experiment_id
  logical :: obs_lai, obs_soilmoist, obs_snowdepth, obs_manage, yasso_year, &
             phydro_debug, yasso_debug, water_debug
  integer :: log_level

contains

  subroutine readctrl_namelist
    ! No-op stub — variables are set directly by the wrapper
  end subroutine readctrl_namelist

END MODULE readctrl_mod
