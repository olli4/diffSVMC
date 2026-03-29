#' Run the full SVMC integration loop
#'
#' Wraps the complete SVMC (Simple Vegetation Model of Carbon) Fortran code:
#' P-hydro photosynthesis-hydraulics, SpaFHy canopy/soil water, YASSO20 soil
#' decomposition, and carbon allocation.  All input data is passed as R arrays
#' (no file I/O).
#'
#' @param nhours Integer, total simulation hours.
#' @param ndays Integer, total simulation days (nhours / 24).
#' @param time_step Numeric, time step in hours (default 1).
#'
#' @param obs_lai Logical, use observed LAI from \code{lai_day}?
#' @param obs_soilmoist Logical, use observed soil moisture from \code{soilmoist_day}?
#' @param obs_snowdepth Logical, use observed snow depth from \code{snowdepth_day}?
#' @param lat,lon Site latitude and longitude.
#'
#' @param conductivity,psi50,b Plant hydraulic parameters.
#' @param alpha,gamma Cost parameters for P-hydro.
#' @param rdark Dark respiration parameter.
#' @param pft_type_code Integer PFT: 1=grass, 2=oat, 0=other.
#'
#' @param soil_depth,max_poros,fc,wp,ksat Soil physical parameters.
#' @param maxpond,n_van,watres,alpha_van,watsat Soil retention parameters.
#' @param wmax,wmaxsnow,hc,w_leaf Canopy water parameters.
#' @param rw,rwmin,gsoil Conductance parameters.
#' @param kmelt,kfreeze,frac_snowliq Snow model parameters.
#' @param zmeas,zground,zo_ground Flow field parameters.
#'
#' @param cratio_resp,cratio_leaf,cratio_root,cratio_biomass Allocation ratios.
#' @param harvest_index Harvest index.
#' @param turnover_cleaf,turnover_croot Turnover rates.
#' @param sla Specific leaf area.
#' @param q10 Q10 temperature coefficient.
#' @param invert_option Allocation inversion: 0=none, 1=cratio_leaf, 2=turnover.
#'
#' @param yasso_totc Initial total soil C (kg/m2).
#' @param yasso_cn_input C:N ratio of input.
#' @param yasso_fract_root Fraction of root input.
#' @param yasso_fract_legacy Fraction of legacy SOC.
#' @param yasso_init_temp Optional initialization temperature for YASSO (deg C).
#'   Defaults to the mean daily temperature derived from \\code{temp_hr}.
#' @param yasso_init_temp_ampl Optional initialization annual temperature
#'   amplitude for YASSO (deg C). Defaults to half the range of daily mean
#'   temperatures derived from \\code{temp_hr}.
#' @param yasso_init_precip Optional initialization daily precipitation for
#'   YASSO (mm/day). Defaults to the mean daily precipitation derived from
#'   \\code{prec_hr}.
#'
#' @param temp_hr Numeric vector (nhours), temperature in K.
#' @param rg_hr Numeric vector (nhours), global radiation in W/m2.
#' @param prec_hr Numeric vector (nhours), precipitation in mm/s.
#' @param vpd_hr Numeric vector (nhours), vapour pressure deficit in Pa.
#' @param pres_hr Numeric vector (nhours), atmospheric pressure in Pa.
#' @param co2_hr Numeric vector (nhours), CO2 as fraction (e.g. 400e-6).
#' @param wind_hr Numeric vector (nhours), wind speed in m/s.
#'
#' @param lai_day Numeric vector (ndays), observed LAI.
#' @param snowdepth_day Numeric vector (ndays), observed snow depth in m.
#' @param soilmoist_day Numeric vector (ndays), observed volumetric soil moisture.
#' @param manage_type Integer vector (ndays), management type codes.
#' @param manage_c_in Numeric vector (ndays), C input from management.
#' @param manage_c_out Numeric vector (ndays), C output from management.
#'
#' @return A list with hourly and daily output arrays.
#' @export
svmc_run <- function(
    nhours, ndays,
    temp_hr, rg_hr, prec_hr, vpd_hr, pres_hr, co2_hr, wind_hr,
    lai_day, snowdepth_day, soilmoist_day,
    # Control
    time_step = 1.0,
    obs_lai = TRUE, obs_soilmoist = FALSE, obs_snowdepth = FALSE,
    lat = 60.295, lon = 22.391,
    # Vegetation
    conductivity = 3e-17, psi50 = -4, b = 2,
    alpha = 0.08, gamma = 1, rdark = 0,
    pft_type_code = 1L,
    # Soil hydro
    soil_depth = 0.6, max_poros = 0.54, fc = 0.40, wp = 0.12, ksat = 2e-6,
    maxpond = 0, n_van = 1.14, watres = 0, alpha_van = 5.92, watsat = 0.68,
    wmax = 0.5, wmaxsnow = 4.5, hc = 0.6, w_leaf = 0.01,
    rw = 0.20, rwmin = 0.02, gsoil = 5e-3,
    kmelt = 2.8934e-05, kfreeze = 5.79e-6, frac_snowliq = 0.05,
    zmeas = 2.0, zground = 0.1, zo_ground = 0.01,
    # Allocation
    cratio_resp = 0.4, cratio_leaf = 0.8, cratio_root = 0.2,
    cratio_biomass = 0.42,
    harvest_index = 0.5,
    turnover_cleaf = 0.41/365, turnover_croot = 0.41/365,
    sla = 10, q10 = 1, invert_option = 0L,
    # Yasso
    yasso_totc = 16, yasso_cn_input = 50,
    yasso_fract_root = 0.5, yasso_fract_legacy = 0,
    yasso_init_temp = NULL,
    yasso_init_temp_ampl = NULL,
    yasso_init_precip = NULL,
    # Management
    manage_type = NULL, manage_c_in = NULL, manage_c_out = NULL) {

  nhours <- as.integer(nhours)
  ndays  <- as.integer(ndays)

  if (nhours != ndays * 24L) {
    stop("nhours must equal ndays * 24 for the hourly-to-daily integration loop")
  }

  if (length(temp_hr) != nhours || length(rg_hr) != nhours ||
      length(prec_hr) != nhours || length(vpd_hr) != nhours ||
      length(pres_hr) != nhours || length(co2_hr) != nhours ||
      length(wind_hr) != nhours) {
    stop("all hourly forcing vectors must have length nhours")
  }

  if (length(lai_day) != ndays) {
    stop("lai_day must have length ndays")
  }

  if (!isTRUE(yasso_fract_root >= 0 && yasso_fract_root <= 1)) {
    stop("yasso_fract_root must be between 0 and 1")
  }
  if (!isTRUE(yasso_fract_legacy >= 0 && yasso_fract_legacy <= 1)) {
    stop("yasso_fract_legacy must be between 0 and 1")
  }

  # Default management arrays
  if (is.null(manage_type)) manage_type <- integer(ndays)
  if (is.null(manage_c_in)) manage_c_in <- numeric(ndays)
  if (is.null(manage_c_out)) manage_c_out <- numeric(ndays)

  # Default observation arrays
  if (is.null(snowdepth_day) || length(snowdepth_day) == 0)
    snowdepth_day <- numeric(ndays)
  if (is.null(soilmoist_day) || length(soilmoist_day) == 0)
    soilmoist_day <- numeric(ndays)

  if (length(snowdepth_day) != ndays || length(soilmoist_day) != ndays ||
      length(manage_type) != ndays || length(manage_c_in) != ndays ||
      length(manage_c_out) != ndays) {
    stop("all daily observation and management vectors must have length ndays")
  }

  if (is.null(yasso_init_temp) || is.null(yasso_init_temp_ampl) || is.null(yasso_init_precip)) {
    temp_day_matrix <- matrix(as.double(temp_hr) - 273.15, ncol = 24, byrow = TRUE)
    prec_day_matrix <- matrix(as.double(prec_hr), ncol = 24, byrow = TRUE)
    daily_temp_c <- rowMeans(temp_day_matrix)
    daily_precip_mm <- rowSums(prec_day_matrix) * 3600

    if (is.null(yasso_init_temp)) {
      yasso_init_temp <- mean(daily_temp_c)
    }
    if (is.null(yasso_init_temp_ampl)) {
      yasso_init_temp_ampl <- 0.5 * (max(daily_temp_c) - min(daily_temp_c))
    }
    if (is.null(yasso_init_precip)) {
      yasso_init_precip <- mean(daily_precip_mm)
    }
  }

  if (!all(is.finite(c(yasso_init_temp, yasso_init_temp_ampl, yasso_init_precip)))) {
    stop("derived YASSO initialization climate must be finite")
  }

  # Pack scalar parameters into arrays for .Fortran (MAX_ARGS=65 limit)
  # iparams(7): nhours, ndays, obs_lai, obs_soilmoist, obs_snowdepth, pft_type_code, invert_option
  iparams <- as.integer(c(nhours, ndays,
                           as.integer(obs_lai), as.integer(obs_soilmoist),
                           as.integer(obs_snowdepth), pft_type_code, invert_option))
  # rparams(48): see svmc_wrapper.f90 for order
  rparams <- as.double(c(
    time_step, lat, lon,
    conductivity, psi50, b, alpha, gamma, rdark,
    soil_depth, max_poros, fc, wp, ksat,
    maxpond, n_van, watres, alpha_van, watsat,
    wmax, wmaxsnow, hc, w_leaf,
    rw, rwmin, gsoil,
    kmelt, kfreeze, frac_snowliq,
    zmeas, zground, zo_ground,
    cratio_resp, cratio_leaf, cratio_root, cratio_biomass,
    harvest_index, turnover_cleaf, turnover_croot,
    sla, q10,
    yasso_totc, yasso_cn_input, yasso_fract_root, yasso_fract_legacy,
    yasso_init_temp, yasso_init_temp_ampl, yasso_init_precip))

  res <- .Fortran("r_svmc_run",
    iparams = iparams,
    rparams = rparams,
    # Hourly forcing
    temp_hr = as.double(temp_hr),
    rg_hr   = as.double(rg_hr),
    prec_hr = as.double(prec_hr),
    vpd_hr  = as.double(vpd_hr),
    pres_hr = as.double(pres_hr),
    co2_hr  = as.double(co2_hr),
    wind_hr = as.double(wind_hr),
    # Daily observations
    lai_day       = as.double(lai_day),
    snowdepth_day = as.double(snowdepth_day),
    soilmoist_day = as.double(soilmoist_day),
    manage_type   = as.integer(manage_type),
    manage_c_in   = as.double(manage_c_in),
    manage_c_out  = as.double(manage_c_out),
    # Hourly outputs (pre-allocated)
    gpp_hr       = double(nhours),
    gs_hr        = double(nhours),
    vcmax_hr     = double(nhours),
    jmax_hr      = double(nhours),
    chi_hr       = double(nhours),
    dpsi_hr      = double(nhours),
    le_hr        = double(nhours),
    tr_hr        = double(nhours),
    soilmoist_hr = double(nhours),
    psi_soil_hr  = double(nhours),
    swe_hr       = double(nhours),
    # Daily outputs (pre-allocated)
    gpp_day      = double(ndays),
    nee_day      = double(ndays),
    npp_day      = double(ndays),
    hetero_resp  = double(ndays),
    auto_resp    = double(ndays),
    lai_alloc    = double(ndays),
    cleaf        = double(ndays),
    croot        = double(ndays),
    cstem        = double(ndays),
    soil_c       = double(ndays),
    above_bio    = double(ndays),
    below_bio    = double(ndays))

  # Return named list of outputs
  list(
    # Hourly
    gpp_hr       = res$gpp_hr,
    gs_hr        = res$gs_hr,
    vcmax_hr     = res$vcmax_hr,
    jmax_hr      = res$jmax_hr,
    chi_hr       = res$chi_hr,
    dpsi_hr      = res$dpsi_hr,
    le_hr        = res$le_hr,
    tr_hr        = res$tr_hr,
    soilmoist_hr = res$soilmoist_hr,
    psi_soil_hr  = res$psi_soil_hr,
    swe_hr       = res$swe_hr,
    # Daily
    gpp_day      = res$gpp_day,
    nee_day      = res$nee_day,
    npp_day      = res$npp_day,
    hetero_resp  = res$hetero_resp,
    auto_resp    = res$auto_resp,
    lai_alloc    = res$lai_alloc,
    cleaf        = res$cleaf,
    croot        = res$croot,
    cstem        = res$cstem,
    soil_c       = res$soil_c,
    above_bio    = res$above_bio,
    below_bio    = res$below_bio
  )
}
