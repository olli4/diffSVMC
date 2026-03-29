! R-callable wrapper subroutines for the SVMC allocation module.
!
! These wrappers flatten Fortran derived types (alloc_para_type,
! management_data_type) into individual scalar arguments so they
! can be called from R via .Fortran().  They also set the module-level
! pft_type global from an integer code.
!
! pft_type codes: 1 = "grass", 2 = "oat", 0 = other

subroutine r_alloc_h2( &
    temp_day, gpp_day, npp_day, leaf_rdark_day, auto_resp, &
    croot, cleaf, cstem, cgrain, &
    litter_cleaf, litter_croot, compost, &
    abovebiomass, belowbiomass, yield_val, &
    lai, grain_fill, &
    cratio_resp, cratio_leaf, cratio_root, cratio_biomass, &
    harvest_index, turnover_cleaf, turnover_croot, &
    sla, q10, invert_option, &
    management_type, management_c_input, management_c_output, &
    management_n_input, management_n_output, &
    pheno_stage, pft_type_code)

  use allocation
  use readvegpara_mod, only: pft_type

  implicit none

  double precision, intent(in)    :: temp_day, gpp_day, leaf_rdark_day
  double precision, intent(inout) :: npp_day, auto_resp
  double precision, intent(inout) :: croot, cleaf, cstem, cgrain
  double precision, intent(inout) :: litter_cleaf, litter_croot, compost
  double precision, intent(inout) :: abovebiomass, belowbiomass, yield_val
  double precision, intent(inout) :: lai, grain_fill
  double precision, intent(inout) :: cratio_resp, cratio_leaf, cratio_root
  double precision, intent(inout) :: cratio_biomass
  double precision, intent(inout) :: harvest_index, turnover_cleaf, turnover_croot
  double precision, intent(inout) :: sla, q10, invert_option
  integer, intent(inout)          :: management_type
  double precision, intent(inout) :: management_c_input, management_c_output
  double precision, intent(inout) :: management_n_input, management_n_output
  integer, intent(inout)          :: pheno_stage
  integer, intent(in)             :: pft_type_code

  type(alloc_para_type) :: ap
  type(management_data_type) :: md

  ! Set pft_type global from integer code
  select case (pft_type_code)
    case (1)
      pft_type = "grass"
    case (2)
      pft_type = "oat"
    case default
      pft_type = "other"
  end select

  ! Pack derived types from scalar arguments
  ap%cratio_resp    = cratio_resp
  ap%cratio_leaf    = cratio_leaf
  ap%cratio_root    = cratio_root
  ap%cratio_biomass = cratio_biomass
  ap%harvest_index  = harvest_index
  ap%turnover_cleaf = turnover_cleaf
  ap%turnover_croot = turnover_croot
  ap%sla            = sla
  ap%q10            = q10
  ap%invert_option  = invert_option

  md%management_type     = management_type
  md%management_c_input  = management_c_input
  md%management_c_output = management_c_output
  md%management_n_input  = management_n_input
  md%management_n_output = management_n_output

  ! Call the original module subroutine
  call alloc_hypothesis_2( &
      temp_day, gpp_day, npp_day, leaf_rdark_day, auto_resp, &
      croot, cleaf, cstem, cgrain, &
      litter_cleaf, litter_croot, compost, &
      abovebiomass, belowbiomass, yield_val, &
      lai, ap, grain_fill, md, pheno_stage)

  ! Unpack modified derived type fields back to scalar arguments
  cratio_resp    = ap%cratio_resp
  cratio_leaf    = ap%cratio_leaf
  cratio_root    = ap%cratio_root
  cratio_biomass = ap%cratio_biomass
  harvest_index  = ap%harvest_index
  turnover_cleaf = ap%turnover_cleaf
  turnover_croot = ap%turnover_croot
  sla            = ap%sla
  q10            = ap%q10
  invert_option  = ap%invert_option

  management_type     = md%management_type
  management_c_input  = md%management_c_input
  management_c_output = md%management_c_output
  management_n_input  = md%management_n_input
  management_n_output = md%management_n_output

end subroutine r_alloc_h2


subroutine r_invert_alloc( &
    delta_lai, temp_day, gpp_day, leaf_rdark_day, &
    litter_cleaf, cleaf, cstem, &
    cratio_resp, cratio_leaf, cratio_root, cratio_biomass, &
    harvest_index, turnover_cleaf, turnover_croot, &
    sla, q10, invert_option, &
    management_type, management_c_input, management_c_output, &
    management_n_input, management_n_output, &
    pheno_stage, pft_type_code)

  use allocation
  use readvegpara_mod, only: pft_type

  implicit none

  double precision, intent(inout) :: delta_lai
  double precision, intent(in)    :: temp_day, gpp_day, leaf_rdark_day
  double precision, intent(inout) :: litter_cleaf, cleaf, cstem
  double precision, intent(inout) :: cratio_resp, cratio_leaf, cratio_root
  double precision, intent(inout) :: cratio_biomass
  double precision, intent(inout) :: harvest_index, turnover_cleaf, turnover_croot
  double precision, intent(inout) :: sla, q10, invert_option
  integer, intent(inout)          :: management_type
  double precision, intent(inout) :: management_c_input, management_c_output
  double precision, intent(inout) :: management_n_input, management_n_output
  integer, intent(in)             :: pheno_stage
  integer, intent(in)             :: pft_type_code

  type(alloc_para_type) :: ap
  type(management_data_type) :: md

  ! Set pft_type global from integer code
  select case (pft_type_code)
    case (1)
      pft_type = "grass"
    case (2)
      pft_type = "oat"
    case default
      pft_type = "other"
  end select

  ! Pack derived types
  ap%cratio_resp    = cratio_resp
  ap%cratio_leaf    = cratio_leaf
  ap%cratio_root    = cratio_root
  ap%cratio_biomass = cratio_biomass
  ap%harvest_index  = harvest_index
  ap%turnover_cleaf = turnover_cleaf
  ap%turnover_croot = turnover_croot
  ap%sla            = sla
  ap%q10            = q10
  ap%invert_option  = invert_option

  md%management_type     = management_type
  md%management_c_input  = management_c_input
  md%management_c_output = management_c_output
  md%management_n_input  = management_n_input
  md%management_n_output = management_n_output

  ! Call the original module subroutine
  call invert_alloc(delta_lai, ap, leaf_rdark_day, temp_day, litter_cleaf, &
                    gpp_day, cleaf, cstem, md, pheno_stage)

  ! Unpack modified derived type fields
  cratio_resp    = ap%cratio_resp
  cratio_leaf    = ap%cratio_leaf
  cratio_root    = ap%cratio_root
  cratio_biomass = ap%cratio_biomass
  harvest_index  = ap%harvest_index
  turnover_cleaf = ap%turnover_cleaf
  turnover_croot = ap%turnover_croot
  sla            = ap%sla
  q10            = ap%q10
  invert_option  = ap%invert_option

  management_type     = md%management_type
  management_c_input  = md%management_c_input
  management_c_output = md%management_c_output
  management_n_input  = md%management_n_input
  management_n_output = md%management_n_output

end subroutine r_invert_alloc
