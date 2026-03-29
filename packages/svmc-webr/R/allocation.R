#' Run the SVMC alloc_hypothesis_2 allocation subroutine
#'
#' Wraps the Fortran \code{alloc_hypothesis_2} subroutine from the SVMC
#' allocation module.  All derived-type members are passed as individual
#' scalar arguments and returned in the result list.
#'
#' @param temp_day  Temperature (exponentially averaged), degrees C.
#' @param gpp_day   GPP (daily average), kg C m-2 s-1.
#' @param leaf_rdark_day  Leaf dark respiration (daily average), kg C m-2 s-1.
#' @param npp_day   NPP (daily average), kg C m-2 s-1 (in/out).
#' @param auto_resp Autotrophic respiration (in/out).
#' @param croot,cleaf,cstem,cgrain Carbon pools (in/out).
#' @param litter_cleaf,litter_croot,compost Litter/compost fluxes (in/out).
#' @param abovebiomass,belowbiomass Biomass pools (in/out).
#' @param yield_val Yield (in/out).
#' @param lai       Leaf area index (in/out).
#' @param grain_fill Grain-filling C flux, kg C m-2 s-1 (in/out).
#' @param cratio_resp,cratio_leaf,cratio_root,cratio_biomass Allocation ratios.
#' @param harvest_index Harvest index.
#' @param turnover_cleaf,turnover_croot Turnover rates.
#' @param sla       Specific leaf area.
#' @param q10       Q10 temperature coefficient.
#' @param invert_option 0 = no inversion, 1 = cratio_leaf, 2 = turnover_leaf.
#' @param management_type Integer management code (0/1/3/4).
#' @param management_c_input,management_c_output C management fluxes.
#' @param management_n_input,management_n_output N management fluxes.
#' @param pheno_stage Phenological stage (1 = growth, 2 = dormancy).
#' @param pft_type_code Plant functional type: 1 = grass, 2 = oat, 0 = other.
#'
#' @return A named list with all in/out arguments after the Fortran call.
#' @export
alloc_hypothesis_2 <- function(
    temp_day, gpp_day, leaf_rdark_day,
    npp_day = 0, auto_resp = 0,
    croot = 0, cleaf = 0, cstem = 0, cgrain = 0,
    litter_cleaf = 0, litter_croot = 0, compost = 0,
    abovebiomass = 0, belowbiomass = 0, yield_val = 0,
    lai = 0, grain_fill = 0,
    cratio_resp = 0.4, cratio_leaf = 0.8, cratio_root = 0.2,
    cratio_biomass = 0.42,
    harvest_index = 0.5,
    turnover_cleaf = 0.41/365, turnover_croot = 0.41/365,
    sla = 10, q10 = 1, invert_option = 0,
    management_type = 0L,
    management_c_input = 0, management_c_output = 0,
    management_n_input = 0, management_n_output = 0,
    pheno_stage = 1L,
    pft_type_code = 0L) {

  .Fortran("r_alloc_h2",
    temp_day         = as.double(temp_day),
    gpp_day          = as.double(gpp_day),
    npp_day          = as.double(npp_day),
    leaf_rdark_day   = as.double(leaf_rdark_day),
    auto_resp        = as.double(auto_resp),
    croot            = as.double(croot),
    cleaf            = as.double(cleaf),
    cstem            = as.double(cstem),
    cgrain           = as.double(cgrain),
    litter_cleaf     = as.double(litter_cleaf),
    litter_croot     = as.double(litter_croot),
    compost          = as.double(compost),
    abovebiomass     = as.double(abovebiomass),
    belowbiomass     = as.double(belowbiomass),
    yield_val        = as.double(yield_val),
    lai              = as.double(lai),
    grain_fill       = as.double(grain_fill),
    cratio_resp      = as.double(cratio_resp),
    cratio_leaf      = as.double(cratio_leaf),
    cratio_root      = as.double(cratio_root),
    cratio_biomass   = as.double(cratio_biomass),
    harvest_index    = as.double(harvest_index),
    turnover_cleaf   = as.double(turnover_cleaf),
    turnover_croot   = as.double(turnover_croot),
    sla              = as.double(sla),
    q10              = as.double(q10),
    invert_option    = as.double(invert_option),
    management_type  = as.integer(management_type),
    management_c_input  = as.double(management_c_input),
    management_c_output = as.double(management_c_output),
    management_n_input  = as.double(management_n_input),
    management_n_output = as.double(management_n_output),
    pheno_stage      = as.integer(pheno_stage),
    pft_type_code    = as.integer(pft_type_code),
    PACKAGE = "SVMCwebr"
  )
}


#' Run the SVMC invert_alloc subroutine
#'
#' Wraps the Fortran \code{invert_alloc} subroutine.  Derives allometric
#' parameters (cratio_leaf or turnover_cleaf) from observed LAI changes.
#'
#' @param delta_lai Change in LAI (in/out).
#' @param temp_day  Temperature (exponentially averaged), degrees C.
#' @param gpp_day   GPP (daily average), kg C m-2 s-1.
#' @param leaf_rdark_day Leaf dark respiration, kg C m-2 s-1.
#' @param litter_cleaf Leaf litter carbon flux (in/out).
#' @param cleaf,cstem Carbon pools (in/out).
#' @param cratio_resp,cratio_leaf,cratio_root,cratio_biomass Allocation ratios.
#' @param harvest_index Harvest index.
#' @param turnover_cleaf,turnover_croot Turnover rates.
#' @param sla       Specific leaf area.
#' @param q10       Q10 temperature coefficient.
#' @param invert_option 1 = derive cratio_leaf, 2 = derive turnover_leaf.
#' @param management_type Integer management code.
#' @param management_c_input,management_c_output C management fluxes.
#' @param management_n_input,management_n_output N management fluxes.
#' @param pheno_stage Phenological stage (1 = active).
#' @param pft_type_code Plant functional type: 1 = grass, 2 = oat, 0 = other.
#'
#' @return A named list with all in/out arguments after the Fortran call.
#' @export
invert_alloc <- function(
    delta_lai,
    temp_day, gpp_day, leaf_rdark_day,
    litter_cleaf = 0, cleaf = 0, cstem = 0,
    cratio_resp = 0.4, cratio_leaf = 0.8, cratio_root = 0.2,
    cratio_biomass = 0.42,
    harvest_index = 0.5,
    turnover_cleaf = 0.41/365, turnover_croot = 0.41/365,
    sla = 10, q10 = 1, invert_option = 1,
    management_type = 0L,
    management_c_input = 0, management_c_output = 0,
    management_n_input = 0, management_n_output = 0,
    pheno_stage = 1L,
    pft_type_code = 0L) {

  .Fortran("r_invert_alloc",
    delta_lai        = as.double(delta_lai),
    temp_day         = as.double(temp_day),
    gpp_day          = as.double(gpp_day),
    leaf_rdark_day   = as.double(leaf_rdark_day),
    litter_cleaf     = as.double(litter_cleaf),
    cleaf            = as.double(cleaf),
    cstem            = as.double(cstem),
    cratio_resp      = as.double(cratio_resp),
    cratio_leaf      = as.double(cratio_leaf),
    cratio_root      = as.double(cratio_root),
    cratio_biomass   = as.double(cratio_biomass),
    harvest_index    = as.double(harvest_index),
    turnover_cleaf   = as.double(turnover_cleaf),
    turnover_croot   = as.double(turnover_croot),
    sla              = as.double(sla),
    q10              = as.double(q10),
    invert_option    = as.double(invert_option),
    management_type  = as.integer(management_type),
    management_c_input  = as.double(management_c_input),
    management_c_output = as.double(management_c_output),
    management_n_input  = as.double(management_n_input),
    management_n_output = as.double(management_n_output),
    pheno_stage      = as.integer(pheno_stage),
    pft_type_code    = as.integer(pft_type_code),
    PACKAGE = "SVMCwebr"
  )
}
