MODULE readctrl_mod

  !*****************************************************************************
  !                                                                            *
  !     This routine reads the user specifications for the current model run.  *
  !                                                                            *
  !     Author: H. Tang                                                        *
  !*****************************************************************************
  !                                                                            *
  !                                                                            *
  ! Variables:                                                                 *
  ! unitcommand          unit connected to file COMMAND                        *
  ! start_date_day       start date of simulation                              *
  ! start_date_hour      start time of simulation                              *
  ! time_step            output time step                                      *
  ! end_date_day         end date of simulation                                *
  ! end_date_hour        end time of simulation                                *
  ! output_directory     directory of output                                   *
  ! output_filename      file name of output                                   *
  !                                                                            *
  !*****************************************************************************

  implicit none

  public :: readctrl_namelist
  
  !Input files/settings

  integer,parameter :: dp=selected_real_kind(P=15)
  
  integer :: start_date_day, start_date_hour, &
             end_date_day, end_date_hour
  integer :: num_sites              ! This can be set by reading input file
  real(8), dimension(1)    :: lat_sites, lon_sites   ! This can be set by reading input file 
  real    :: time_step, time_step_output
  character(len=256)  :: output_dir, input_dir
  character(len=256)  :: sites_name, experiment_id           ! need to be allocable.
  logical :: obs_lai, obs_soilmoist, obs_snowdepth, obs_manage, yasso_year, &
             phydro_debug, yasso_debug, water_debug
  integer :: log_level

contains
  !------------------------------------------------------
  subroutine readctrl_namelist
    logical :: old
    integer :: readerror
    integer,parameter :: unitcommand=1

    namelist /ctrl_namelist/ &
    start_date_day, &
    start_date_hour, &
    time_step, &
    end_date_day, &
    end_date_hour, &
    num_sites, & 
    sites_name, &
    input_dir, &
    output_dir, &
    time_step_output, &
    obs_lai, &
    obs_soilmoist, &
    obs_snowdepth, &
    obs_manage,  &
    yasso_year,  &
    phydro_debug, &
    yasso_debug, & 
    water_debug, &
    log_level, &
    experiment_id

    old=.false.
    
    ! Presetting namelist command
    start_date_day  =20190101 !20190601
    start_date_hour =000000
    time_step       =1               ! hours
    end_date_day    =20191231 !
    end_date_hour   =000000
    num_sites       =1                 ! number of sites, should also be read from input file?
    sites_name      ='Qvidja'
    ! lon_sites     = (/ /)            ! longitude of sites (not needed, can be well defined input file)
    ! lat_sites     = (/ /)            ! latitude of sites (not needed, can be well defined by input file) 
    input_dir      ='../../SVMC/data/input'  
    output_dir    ='../data/output'       
    time_step_output    =1.0
    obs_lai=.true.
    obs_soilmoist=.false.
    obs_snowdepth=.false.
    obs_manage=.false.
    yasso_year=.false.
    phydro_debug=.false.
    yasso_debug=.false.
    water_debug=.false.
    log_level=1
    experiment_id="test"

    ! Reading namelist
    open(unitcommand, file='./ctrl_namelist', status='old', form='formatted', err=999)
    read(unitcommand, ctrl_namelist, iostat=readerror)
    close(unitcommand)
    
    print *, "experiment_id= ", experiment_id

    if (time_step.eq.0) then
      write(*,*) ' #### MODEL ERROR! TIME STEP MUST    #### '
      write(*,*) ' #### NOT BE ZERO                             #### '
      write(*,*) ' #### CHANGE INPUT IN FILE COMMAND.           #### '
      stop
    endif

    return

999   write(*,*) ' #### MODEL ERROR! FILE "ctrl_namelist"    #### '
    write(*,*) ' #### CANNOT BE OPENED IN THE DIRECTORY       #### '
    stop

  end subroutine readctrl_namelist
                   
end module readctrl_mod
