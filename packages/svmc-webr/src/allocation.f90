! SVMC allocation module — R-package copy
!
! Provenance: vendor/SVMC/src/allocation.f90
! Modifications for R package compatibility:
!   1. litter_cleaf, litter_croot, compost in alloc_hypothesis_2 changed from
!      real to real(8). The vendor build uses -freal-4-real-8 so these are
!      effectively double precision in the reference model.
!   2. readalloc_namelist subroutine removed (does file I/O incompatible with
!      WebR; parameters are passed via R arguments instead).
!
! No numerical changes relative to the vendor build with -freal-4-real-8.

module allocation

use readvegpara_mod

implicit none

   type, public :: alloc_para_type
   !----------------------------
   ! allometric parameters
   !-----------------------------
     real(8) :: cratio_resp         ! fraction of maintenance respiration (1/m2/s) at 20 degree
     real(8) :: cratio_leaf         ! carbon ratio of leaf to npp
     real(8) :: cratio_root         ! carbon ratio of root to npp
     real(8) :: cratio_biomass      ! carbon ratio of biomass
     real(8) :: harvest_index       !
     real(8) :: turnover_cleaf      ! turnover rate of leaf at 20 degree
     real(8) :: turnover_croot      ! turnover rate of root at 20 degree
     real(8) :: sla                 ! specific leaf area,
     real(8) :: q10                 ! Q10 temperature coefficient (https://en.wikipedia.org/wiki/Q10_(temperature_coefficient))
     real(8) :: invert_option       ! 0: no inversion from LAI
                                    ! 1: inversion for cratio_leaf
                                    ! 2: inversion for turnover_leaf

   end type alloc_para_type

   type, public :: management_data_type
   !----------------------------
   ! management parameters
   !-----------------------------
     integer :: management_type          ! management types
     real(8) :: management_c_input       !
     real(8) :: management_c_output      !
     real(8) :: management_n_input       !
     real(8) :: management_n_output      !
   end type management_data_type

   public alloc_hypothesis_1    ! estimate litter input from gpp directly, fixed ratio
   public alloc_hypothesis_2    ! estimate litter input from npp -> above- & below- ground biomass, only management
   public readalloc_namelist    ! no-op stub (parameters set by wrapper)

contains

   subroutine readalloc_namelist(alloc_para)
     type(alloc_para_type), intent(inout) :: alloc_para
     ! No-op stub — parameters set directly by the R wrapper
   end subroutine readalloc_namelist

   subroutine alloc_hypothesis_1(gpp_day, npp_day, litter_cleaf, litter_croot, alloc_para)
      real(8), intent(in)    :: gpp_day      ! gpp (daily average),  kg C m-2 s-1
      real(8), intent(inout) :: npp_day      ! npp (daily average),  kg C m-2 s-1
      real(8), intent(inout) :: litter_cleaf ! carbon input with "leaf" composition per day
      real(8), intent(inout) :: litter_croot ! carbon input with "root" composition per day
      type(alloc_para_type), intent(inout) :: alloc_para   ! allometric parameters

      alloc_para%cratio_resp = 0.5
      alloc_para%cratio_leaf =0.5
      alloc_para%harvest_index = 0.5

      litter_cleaf= gpp_day * (1-alloc_para%cratio_resp) * (1-alloc_para%harvest_index) * &
                               alloc_para%cratio_leaf * 3600 * 24
      litter_croot= gpp_day * (1-alloc_para%cratio_resp) * (1-alloc_para%harvest_index) * &
                              (1-alloc_para%cratio_leaf) * 3600 * 24

   end subroutine alloc_hypothesis_1

   subroutine alloc_hypothesis_2(temp_day, gpp_day, npp_day, leaf_rdark_day, auto_resp, croot, cleaf, cstem, cgrain, &
                                  litter_cleaf, litter_croot, compost, abovebiomass, belowbiomass, yield, &
                                  lai, alloc_para, grain_fill, manage_data, pheno_stage)

      real(8), intent(in)    :: temp_day     ! temperature (exponentially averaged),  celcius degree
      real(8), intent(in)    :: gpp_day      ! gpp (daily average),  kg C m-2 s-1
      real(8), intent(in)    :: leaf_rdark_day    ! leaf dark respiration (daily average), kg C m-2 s-1

      real(8), intent(inout) :: npp_day      ! npp (daily average),  kg C m-2 s-1
      real(8), intent(inout) :: auto_resp      ! npp (daily average),  kg C m-2 s-1
      real(8), intent(inout) :: croot        ! root carbon
      real(8), intent(inout) :: cleaf        ! leaf carbon
      real(8), intent(inout) :: cstem        ! leaf carbon
      real(8), intent(inout) :: cgrain       ! grain carbon
      ! Modified: real -> real(8) to match vendor -freal-4-real-8 promotion
      real(8), intent(inout) :: litter_cleaf ! carbon input with "leaf" composition per day
      real(8), intent(inout) :: litter_croot ! carbon input with "root" composition per day
      real(8), intent(inout) :: compost      ! manure input (kg C m-2 s-1)
      real(8), intent(inout) :: lai          ! leaf area index, allometric
      real(8), intent(inout) :: abovebiomass ! Abovegroud biomass (kg dry mass m-2)
      real(8), intent(inout) :: belowbiomass ! Abovegroud biomass (kg dry mass m-2)
      real(8), intent(inout) :: yield        ! yield
      real(8), intent(inout) :: grain_fill   ! grain-filling C flux, kg C m-2 s-1
      type(alloc_para_type), intent(inout) :: alloc_para   ! allometric parameters
      type(management_data_type), intent(inout) :: manage_data   ! allometric parameters
      integer, intent(inout)    :: pheno_stage

      ! local
      real(8) :: litter_cstem                                  ! carbon input with "stem" composition per day
      real(8) :: gr_resp_leaf, gr_resp_stem, gr_resp_root, gr_resp_grain      ! growth respiration, kg C m-2 s-1


      if (pheno_stage .eq. 1) then

         gr_resp_leaf=max(0.0d0, (gpp_day * alloc_para%cratio_leaf - leaf_rdark_day)*3600*24)*0.11d0
         gr_resp_stem=max((gpp_day * (1-alloc_para%cratio_leaf-alloc_para%cratio_root) &
                                  - cstem * (alloc_para%cratio_resp * alloc_para%q10 ** ((temp_day - 20)/10)))*3600*24,0.0d0)*0.11d0
         gr_resp_root=max((gpp_day * alloc_para%cratio_root - grain_fill  &
                                  - croot * (alloc_para%cratio_resp * alloc_para%q10 ** ((temp_day - 20)/10)))*3600*24,0.0d0)*0.11d0
         if (pft_type=="oat") then
            gr_resp_grain=max((grain_fill - cgrain*(0.1*alloc_para%cratio_resp * alloc_para%q10 ** ((temp_day - 20)/10)))*3600*24, &
                              0.0d0)*0.11d0
         else
            gr_resp_grain=0.0d0
         end if

         npp_day = (gpp_day - (croot + cstem) * (alloc_para%cratio_resp * alloc_para%q10 ** ((temp_day - 20)/10)) &
                            - cgrain * (0.1 * alloc_para%cratio_resp * alloc_para%q10 ** ((temp_day - 20)/10)) &
                            - leaf_rdark_day )* 3600 * 24
         auto_resp = ((croot + cstem) * (alloc_para%cratio_resp * alloc_para%q10 ** ((temp_day - 20)/10)) &
                            + cgrain * (0.1 * alloc_para%cratio_resp * alloc_para%q10 ** ((temp_day - 20)/10)) &
                            + leaf_rdark_day )* 3600 * 24  &
                            + gr_resp_leaf &
                            + gr_resp_stem &
                            + gr_resp_root &
                            + gr_resp_grain

         if (alloc_para%invert_option .eq. 0) then
           litter_cleaf=cleaf * alloc_para%turnover_cleaf * alloc_para%q10 ** ((temp_day  - 20)/10)
         end if
         litter_cstem=cstem * alloc_para%turnover_croot * alloc_para%q10 ** ((temp_day  - 20)/10)
         litter_croot=croot * alloc_para%turnover_croot * alloc_para%q10 ** ((temp_day  - 20)/10)
         compost=0.0d0

         if (alloc_para%invert_option .eq. 0) then
            cleaf   = cleaf + gpp_day * alloc_para%cratio_leaf * 3600 * 24  - litter_cleaf - leaf_rdark_day*3600*24 - gr_resp_leaf
         end if

         cstem   = cstem + gpp_day * (1-alloc_para%cratio_leaf-alloc_para%cratio_root) *3600 *24 &
                     - cstem * (alloc_para%cratio_resp * alloc_para%q10 ** ((temp_day - 20)/10))*3600*24 &
                     - litter_cstem &
                     - gr_resp_stem
         croot   = max(0.0d0, croot + (gpp_day * alloc_para%cratio_root - grain_fill) * 3600 * 24 - litter_croot &
                     - croot * (alloc_para%cratio_resp * alloc_para%q10 ** ((temp_day - 20)/10))*3600*24 &
                     - gr_resp_root)

         if (pft_type=="oat") then
             cgrain  = cgrain + grain_fill*3600*24  &
                     - cgrain * (0.1*alloc_para%cratio_resp * alloc_para%q10 ** ((temp_day - 20)/10))*3600*24 &
                     - gr_resp_grain
         else
            cgrain=0.0d0
         end if

         if (manage_data%management_type .eq. 1) then
            if (pft_type=="grass") then
               if (alloc_para%invert_option .eq. 0) then
                  cleaf = cleaf - manage_data%management_c_output*3600*24*cleaf/(cleaf+cstem)
               end if
               cstem = cstem - manage_data%management_c_output*3600*24*cstem/(cleaf+cstem)
            else if (pft_type=="oat") then
                  litter_croot=croot
                  litter_cleaf=cleaf
                  litter_cstem=cstem
                  croot=0.0d0
                  cleaf=0.0d0
                  cstem=0.0d0
                  cgrain=0.0d0
                  npp_day=0.0d0
                  auto_resp=0.0d0
            end if

         else if (manage_data%management_type .eq. 3) then
            if (alloc_para%invert_option .eq. 0) then
              cleaf = cleaf - manage_data%management_c_output*3600*24*cleaf/(cleaf+cstem)
            end if
            cstem = cstem - manage_data%management_c_output*3600*24*cstem/(cleaf+cstem)
            compost= manage_data%management_c_input*3600*24
         else if (manage_data%management_type .eq. 4) then
            compost= manage_data%management_c_input*3600*24
         end if

         litter_cleaf= litter_cleaf + litter_cstem

      else if (pheno_stage .eq. 2) then
         litter_croot=croot
         litter_cleaf=cleaf
         litter_cstem=cstem
         croot=0.0d0
         cleaf=0.0d0
         cstem=0.0d0
         npp_day=0.0d0
         auto_resp=0.0d0
         pheno_stage=1
      end if

      abovebiomass = (cleaf+cstem+cgrain)/alloc_para%cratio_biomass
      belowbiomass = croot/alloc_para%cratio_biomass
      lai          = cleaf/alloc_para%cratio_biomass * alloc_para%sla

   end subroutine alloc_hypothesis_2

   subroutine invert_alloc(delta_lai, alloc_para, leaf_rdark_day, temp_day, litter_cleaf, gpp_day, cleaf, &
                            cstem, manage_data, pheno_stage)

      real(8), intent(in)    :: temp_day
      real(8), intent(in)    :: gpp_day
      real(8), intent(in)    :: leaf_rdark_day
      real(8), intent(inout) :: cleaf
      real(8), intent(inout) :: cstem
      real(8), intent(inout) :: litter_cleaf
      real(8), intent(inout) :: delta_lai
      type(alloc_para_type), intent(inout) :: alloc_para
      type(management_data_type), intent(inout) :: manage_data
      integer, intent(in)    :: pheno_stage

      ! local
      real(8) :: delta_cleaf
      real(8) :: gr_resp_leaf

      if ( pheno_stage .eq. 1) then

         delta_cleaf = delta_lai/alloc_para%sla * alloc_para%cratio_biomass

         gr_resp_leaf=max(0.0d0, (gpp_day * alloc_para%cratio_leaf - leaf_rdark_day)*3600*24)*0.11d0

         if (alloc_para%invert_option .eq. 1) then
            litter_cleaf = cleaf * alloc_para%turnover_cleaf * alloc_para%q10 ** ((temp_day  - 20)/10)
            if (gpp_day .gt. 0.2d-8) then
               if (manage_data%management_type .eq. 1) then
                  if (pft_type=="grass") then
                     alloc_para%cratio_leaf = (delta_cleaf + litter_cleaf + leaf_rdark_day * 3600*24 + gr_resp_leaf &
                          + manage_data%management_c_output*3600*24*cleaf/(cleaf+cstem))/3600/24/gpp_day
                  else
                     alloc_para%cratio_leaf = (delta_cleaf + litter_cleaf + leaf_rdark_day * 3600*24 + gr_resp_leaf &
                           )/3600/24/gpp_day
                  end if

               else if (manage_data%management_type .eq. 3) then
                  if (pft_type=="grass") then
                     alloc_para%cratio_leaf = (delta_cleaf + litter_cleaf + leaf_rdark_day*3600*24 + gr_resp_leaf &
                        + manage_data%management_c_output*3600*24*cleaf/(cleaf+cstem))/3600/24/gpp_day
                  else
                     alloc_para%cratio_leaf = (delta_cleaf + litter_cleaf + leaf_rdark_day*3600*24 + gr_resp_leaf &
                          )/3600/24/gpp_day
                  end if

               else
                  alloc_para%cratio_leaf = (delta_cleaf + litter_cleaf + leaf_rdark_day*3600*24 + gr_resp_leaf)/3600/24/gpp_day
               end if

               alloc_para%cratio_leaf = min(0.9d0, max(0.1d0, alloc_para%cratio_leaf))
               alloc_para%cratio_root = 1-alloc_para%cratio_leaf
            end if
            cleaf=max(0.0d0, cleaf + delta_cleaf)

         else if (alloc_para%invert_option .eq. 2) then
            if (cleaf .gt. 0.00001d0) then
               if (manage_data%management_type .eq. 1) then
                  if (pft_type=="grass") then
                     alloc_para%turnover_cleaf = (gpp_day * alloc_para%cratio_leaf * 3600 * 24 - delta_cleaf - &
                           manage_data%management_c_output*3600*24*cleaf/(cleaf+cstem) - leaf_rdark_day*3600*24 - gr_resp_leaf)/ &
                              cleaf/(alloc_para%q10 ** ((temp_day  - 20)/10))
                  else
                     alloc_para%turnover_cleaf = (gpp_day * alloc_para%cratio_leaf * 3600 * 24 - delta_cleaf - &
                           leaf_rdark_day*3600*24 - gr_resp_leaf)/ &
                              cleaf/(alloc_para%q10 ** ((temp_day  - 20)/10))
                  end if

               else if (manage_data%management_type .eq. 3) then
                  if (pft_type=="grass") then
                      alloc_para%turnover_cleaf = (gpp_day * alloc_para%cratio_leaf * 3600 * 24 - delta_cleaf - &
                        manage_data%management_c_output*3600*24*cleaf/(cleaf+cstem) - leaf_rdark_day*3600*24 - gr_resp_leaf)/ &
                           cleaf/(alloc_para%q10 ** ((temp_day  - 20)/10))
                  else
                      alloc_para%turnover_cleaf = (gpp_day * alloc_para%cratio_leaf * 3600 * 24 - delta_cleaf - &
                           leaf_rdark_day*3600*24 - gr_resp_leaf)/ &
                              cleaf/(alloc_para%q10 ** ((temp_day  - 20)/10))
                  end if

               else
                  alloc_para%turnover_cleaf = (gpp_day * alloc_para%cratio_leaf * 3600 * 24 - delta_cleaf - &
                         leaf_rdark_day*3600*24 - gr_resp_leaf)/ &
                           cleaf/(alloc_para%q10 ** ((temp_day  - 20)/10))
               end if

            else
               cleaf=0.0d0
            end if
            litter_cleaf= cleaf * alloc_para%turnover_cleaf * alloc_para%q10 ** ((temp_day  - 20)/10)
            cleaf=max(0.0d0,cleaf + delta_cleaf)
         end if

      end if

   end subroutine invert_alloc

end module allocation
