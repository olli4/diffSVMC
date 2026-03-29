! Minimal stub for readvegpara_mod — provides only the declarations
! that allocation.f90 requires (pft_type character variable).
! The full readvegpara_mod.f90 contains namelist I/O routines that
! are incompatible with WebR's limited filesystem.
!
! Provenance: derived from vendor/SVMC/src/readvegpara_mod.f90

MODULE readvegpara_mod

  implicit none

  character(len=256), public :: pft_type = ""

END MODULE readvegpara_mod
