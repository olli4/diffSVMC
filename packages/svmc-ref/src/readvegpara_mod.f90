MODULE readvegpara_mod

  implicit none

  ! !PUBLIC TYPES:
  ! Define type for plant hydraulic parameters
  
  !? Do we need types or just have variables separately
  type, public :: par_plant_type
    !!**** Parameters for P-hydro model
    ! Plant hydraulic parameters
    real(8) :: conductivity     ! Leaf conductivity (m) (for stem, this could be Ks*HV/Height)
    real(8) :: psi50             ! Leaf P50 (Mpa)
    real(8) :: b                     ! Slope of leaf vulnerability curve 
  end type par_plant_type

  type, public :: par_cost_type
    ! A list of cost parameters
    real(8) :: alpha  !cost of Jmax
    real(8) :: gamma    !cost of hydraulic repair
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

  ! constants needed by p-hydro

  real(8)            :: kphio=0.087182     ! Apparent quantum yield efficiency (unitless). default: 0.087182 
  real(8)            :: k=0.5              ! dimensionless constant, assigned a generic value of 0.5, Beer's law.
                                           ! canopy light attenuation parameter (-)
                                          ! light absorption by LAI (according to Qiao et al. 2020)
  real(8)            :: c_molmass =12.0107  ! molecular mass of carbon (g/mol)
  real(8)            :: h2o_molmass =18.01528  ! molecular mass of carbon (g/mol)
  
  character(len=256), public  :: opt_hypothesis, pft_type        ! character, Either "Lc" or "PM"
  integer  :: num_pft
  real(8)  :: conductivity, psi50, b, alpha, gamma, rdark

contains

  subroutine readvegpara_namelist
    implicit none
    
    logical :: old
    integer :: readerror
    integer,parameter :: unitvegpara=2

    namelist /veg_namelist/ &
      num_pft, &
      pft_type, &
      conductivity, &
      psi50, &
      b, &
      alpha, &
      gamma, &
      opt_hypothesis, &
      rdark

    old=.false.
    num_pft=1
    pft_type="grass"
    
    ! Presetting namelist command
    !--- Parameters for P-hydro model
    ! Plant hydraulic parameters
    conductivity=3e-17     ! Leaf conductivity (m) (for stem, this could be Ks*HV/Height)
    psi50 = -4             ! Leaf P50 (Mpa), default is -2
    b=2                  ! Slope of leaf vulnerability curve, default is 2 

    ! A list of cost parameters
    alpha=0.08     !cost of Jmax, default: 0.1
    gamma=1     !cost of hydraulic repair, default: 1

    opt_hypothesis= 'PM'          ! character, Either "LC" or "PM"    

    rdark=0.0                   ! eqivalent to "br" in Josh et al. 2022 
  
   ! Reading namelist
    open(unitvegpara, file='./veg_namelist', status='old', form='formatted', err=999)
    read(unitvegpara,veg_namelist,iostat=readerror)
    close(unitvegpara)

    print *, "rdark= ", rdark

    return

999   write(*,*) ' #### MODEL ERROR! FILE "veg_namelist"    #### '
    write(*,*) ' #### CANNOT BE OPENED IN THE DIRECTORY       #### '
    stop

  end subroutine readvegpara_namelist

                                     
end module readvegpara_mod
