MODULE time_management

!**********************************************************************
! Based on the function from FLEXPART
! Copyright 1998,1999,2000,2001,2002,2005,2007,2008,2009,2010         *
! Andreas Stohl, Petra Seibert, A. Frank, Gerhard Wotawa,             *
! Caroline Forster, Sabine Eckhardt, John Burkhart, Harald Sodemann   *
!**********************************************************************
  use readctrl_mod

  implicit none

  public :: juldate
  public :: isleap

Contains  

  function juldate(yyyymmdd,hhmiss)
  !*****************************************************************************
  !                                                                            *
  !     Calculates the Julian date                                             *
  !                                                                            *
  !     AUTHOR: Andreas Stohl (15 October 1993)                                *
  !                                                                            *
  !     Variables:                                                             *
  !     dd             Day                                                     *
  !     hh             Hour                                                    *
  !     hhmiss         Hour, minute + second                                   *
  !     ja,jm,jy       help variables                                          *
  !     juldate        Julian Date                                             *
  !     julday         help variable                                           *
  !     mi             Minute                                                  *
  !     mm             Month                                                   *
  !     ss             Second                                                  *
  !     yyyy           Year                                                    *
  !     yyyymmddhh     Date and Time                                           *
  !                                                                            *
  !     Constants:                                                             *
  !     igreg          help constant                                           *
  !                                                                            *
  !*****************************************************************************
    integer           :: yyyymmdd,yyyy,mm,dd,hh,mi,ss,hhmiss
    integer           :: julday,jy,jm,ja
    integer,parameter :: igreg=15+31*(10+12*1582)
    real(kind=dp)     :: juldate

    yyyy=yyyymmdd/10000
    mm=(yyyymmdd-10000*yyyy)/100
    dd=yyyymmdd-10000*yyyy-100*mm
    hh=hhmiss/10000
    mi=(hhmiss-10000*hh)/100
    ss=hhmiss-10000*hh-100*mi

    if (yyyy.eq.0) then
      print*, 'there is no year zero.'
      stop
    end if
  
    if (yyyy.lt.0) yyyy=yyyy+1
    if (mm.gt.2) then
      jy=yyyy
      jm=mm+1
    else
      jy=yyyy-1
      jm=mm+13
    endif
  
    julday=int(365.25*jy)+int(30.6001*jm)+dd+1720995
    if (dd+31*(mm+12*yyyy).ge.igreg) then
      ja=int(0.01*jy)
      julday=julday+2-ja+int(0.25*ja)
    endif

    juldate=real(julday,kind=dp)   + real(hh,kind=dp)/24._dp + &
       real(mi,kind=dp)/1440._dp  + real(ss,kind=dp)/86400._dp

  end function juldate

  function isleap(yyyy)
    integer           :: yyyy    ! year.
    logical           :: isleap  ! true for leap year.

    if(mod(yyyy,100)/=0.AND.mod(yyyy,4)==0)then
      isleap=.true.
    elseif(mod(yyyy,400)==0)then
      isleap=.true.
    else
      isleap=.false.
    endif

  end function isleap

END MODULE time_management
