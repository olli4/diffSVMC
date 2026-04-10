import {
  jit,
  prepareRunIntegrationInputs,
  runIntegration,
  tree,
} from "@diffsvmc/svmc-js";

export const YASSO_PARAM = [
  0.51, 5.19, 0.13, 0.1, 0.5, 0.0, 1.0, 1.0, 0.99, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.163, 0.0, -0.0, 0.0, 0.0, 0.0,
  0.0, 0.158, -0.002, 0.17, -0.005, 0.067, -0.0, -1.44,
  -2.0, -6.9, 0.0042, 0.0015, -2.55, 1.24, 0.25,
];

export const JAX_SOIL_DEFAULTS = {
  n_van: 1.14,
  watres: 0.0,
  alpha_van: 5.92,
  maxpond: 0.0,
  wmax: 0.5,
  wmaxsnow: 4.5,
  kmelt: 2.8934e-5,
  kfreeze: 5.79e-6,
  frac_snowliq: 0.05,
  gsoil: 5.0e-3,
  hc: 0.6,
  w_leaf: 0.01,
  rw: 0.2,
  rwmin: 0.02,
  zmeas: 2.0,
  zground: 0.1,
  zo_ground: 0.01,
  yasso_tempr_c: 5.4,
  yasso_precip_day: 1.87,
  yasso_tempr_ampl: 20.0,
};

const runIntegrationJitted = jit((inputs) => runIntegration(inputs));

export function chunkHourly(values, ndays) {
  return Array.from({ length: ndays }, (_, day) => values.slice(day * 24, (day + 1) * 24));
}

export function sliceReferenceData(reference, ndays) {
  const cappedDays = Math.max(1, Math.min(ndays, reference.site.ndays));
  const nhours = cappedDays * 24;
  return {
    nhours,
    ndays: cappedDays,
    hourly: {
      temp_hr: reference.hourly.temp_hr.slice(0, nhours),
      rg_hr: reference.hourly.rg_hr.slice(0, nhours),
      prec_hr: reference.hourly.prec_hr.slice(0, nhours),
      vpd_hr: reference.hourly.vpd_hr.slice(0, nhours),
      pres_hr: reference.hourly.pres_hr.slice(0, nhours),
      co2_hr: reference.hourly.co2_hr.slice(0, nhours),
      wind_hr: reference.hourly.wind_hr.slice(0, nhours),
      timestamps: reference.hourly.timestamps.slice(0, nhours),
    },
    daily: {
      dates: reference.daily.dates.slice(0, cappedDays),
      lai_day: reference.daily.lai_day.slice(0, cappedDays),
      snowdepth_day: reference.daily.snowdepth_day.slice(0, cappedDays),
      soilmoist_day: reference.daily.soilmoist_day.slice(0, cappedDays),
      manage_type: reference.daily.manage_type.slice(0, cappedDays),
      manage_c_in: reference.daily.manage_c_in.slice(0, cappedDays),
      manage_c_out: reference.daily.manage_c_out.slice(0, cappedDays),
    },
    defaults: reference.defaults,
  };
}

export function getDefaultWebsiteControls(referenceDefaults) {
  return {
    pftTypeCode: referenceDefaults.pft_type_code,
    conductivity: referenceDefaults.conductivity,
    psi50: referenceDefaults.psi50,
    bParam: referenceDefaults.b,
    alpha: referenceDefaults.alpha,
    gamma: referenceDefaults.gamma,
    rdark: referenceDefaults.rdark,
    soilDepth: referenceDefaults.soil_depth,
    maxPoros: referenceDefaults.max_poros,
    fc: referenceDefaults.fc,
    wp: referenceDefaults.wp,
    ksat: referenceDefaults.ksat,
    cratioResp: referenceDefaults.cratio_resp,
    cratioLeaf: referenceDefaults.cratio_leaf,
    cratioRoot: referenceDefaults.cratio_root,
    cratioBiomass: referenceDefaults.cratio_biomass,
    harvestIndex: referenceDefaults.harvest_index,
    turnoverCleaf: referenceDefaults.turnover_cleaf,
    turnoverCroot: referenceDefaults.turnover_croot,
    sla: referenceDefaults.sla,
    q10: referenceDefaults.q10,
    yassoTotc: referenceDefaults.yasso_totc,
    yassoCnInput: referenceDefaults.yasso_cn_input,
    yassoFractRoot: referenceDefaults.yasso_fract_root,
    yassoFractLegacy: referenceDefaults.yasso_fract_legacy,
  };
}

export function buildJaxInputs(slice, controls) {
  return {
    hourly_temp: chunkHourly(slice.hourly.temp_hr, slice.ndays),
    hourly_rg: chunkHourly(slice.hourly.rg_hr, slice.ndays),
    hourly_prec: chunkHourly(slice.hourly.prec_hr, slice.ndays),
    hourly_vpd: chunkHourly(slice.hourly.vpd_hr, slice.ndays),
    hourly_pres: chunkHourly(slice.hourly.pres_hr, slice.ndays),
    hourly_co2: chunkHourly(slice.hourly.co2_hr, slice.ndays),
    hourly_wind: chunkHourly(slice.hourly.wind_hr, slice.ndays),
    daily_lai: slice.daily.lai_day,
    daily_manage_type: slice.daily.manage_type.map((value) => Number(value)),
    daily_manage_c_in: slice.daily.manage_c_in,
    daily_manage_c_out: slice.daily.manage_c_out,
    conductivity: controls.conductivity,
    psi50: controls.psi50,
    b_param: controls.bParam,
    alpha_cost: controls.alpha,
    gamma_cost: controls.gamma,
    rdark: controls.rdark,
    soil_depth: controls.soilDepth,
    max_poros: controls.maxPoros,
    fc: controls.fc,
    wp: controls.wp,
    ksat: controls.ksat,
    n_van: JAX_SOIL_DEFAULTS.n_van,
    watres: JAX_SOIL_DEFAULTS.watres,
    alpha_van: JAX_SOIL_DEFAULTS.alpha_van,
    watsat: controls.maxPoros,
    maxpond: JAX_SOIL_DEFAULTS.maxpond,
    wmax: JAX_SOIL_DEFAULTS.wmax,
    wmaxsnow: JAX_SOIL_DEFAULTS.wmaxsnow,
    kmelt: JAX_SOIL_DEFAULTS.kmelt,
    kfreeze: JAX_SOIL_DEFAULTS.kfreeze,
    frac_snowliq: JAX_SOIL_DEFAULTS.frac_snowliq,
    gsoil: JAX_SOIL_DEFAULTS.gsoil,
    hc: JAX_SOIL_DEFAULTS.hc,
    w_leaf: JAX_SOIL_DEFAULTS.w_leaf,
    rw: JAX_SOIL_DEFAULTS.rw,
    rwmin: JAX_SOIL_DEFAULTS.rwmin,
    zmeas: JAX_SOIL_DEFAULTS.zmeas,
    zground: JAX_SOIL_DEFAULTS.zground,
    zo_ground: JAX_SOIL_DEFAULTS.zo_ground,
    cratio_resp: controls.cratioResp,
    cratio_leaf: controls.cratioLeaf,
    cratio_root: controls.cratioRoot,
    cratio_biomass: controls.cratioBiomass,
    harvest_index: controls.harvestIndex,
    turnover_cleaf: controls.turnoverCleaf,
    turnover_croot: controls.turnoverCroot,
    sla: controls.sla,
    q10: controls.q10,
    invert_option: slice.defaults.invert_option,
    pft_is_oat: controls.pftTypeCode === 2 ? 1.0 : 0.0,
    yasso_param: YASSO_PARAM,
    yasso_totc: controls.yassoTotc,
    yasso_cn_input: controls.yassoCnInput,
    yasso_fract_root: controls.yassoFractRoot,
    yasso_fract_legacy: controls.yassoFractLegacy,
    yasso_tempr_c: JAX_SOIL_DEFAULTS.yasso_tempr_c,
    yasso_precip_day: JAX_SOIL_DEFAULTS.yasso_precip_day,
    yasso_tempr_ampl: JAX_SOIL_DEFAULTS.yasso_tempr_ampl,
  };
}

export function buildWebsiteJaxInputLike(reference, ndays, overrides = {}) {
  const slice = sliceReferenceData(reference, ndays);
  const controls = {
    ...getDefaultWebsiteControls(reference.defaults),
    ...overrides,
  };
  return buildJaxInputs(slice, controls);
}

export async function extractDailyOutputs(dailyOutputs) {
  const [
    gpp,
    nee,
    heteroResp,
    autoResp,
    cleaf,
    croot,
    cstem,
    socTotal,
    laiAlloc,
  ] = await Promise.all([
    dailyOutputs.gpp_avg.jsAsync(),
    dailyOutputs.nee.jsAsync(),
    dailyOutputs.hetero_resp.jsAsync(),
    dailyOutputs.auto_resp.jsAsync(),
    dailyOutputs.cleaf.jsAsync(),
    dailyOutputs.croot.jsAsync(),
    dailyOutputs.cstem.jsAsync(),
    dailyOutputs.soc_total.jsAsync(),
    dailyOutputs.lai_alloc.jsAsync(),
  ]);

  return {
    gpp: Array.from(gpp),
    nee: Array.from(nee),
    heteroResp: Array.from(heteroResp),
    autoResp: Array.from(autoResp),
    cleaf: Array.from(cleaf),
    croot: Array.from(croot),
    cstem: Array.from(cstem),
    socTotal: Array.from(socTotal),
    laiAlloc: Array.from(laiAlloc),
  };
}

export async function runWebsiteJaxCore(inputLike, options = {}) {
  const { useJit = true, logger } = options;
  let preparedInputs = null;
  let finalCarry = null;
  let dailyOutputs = null;

  try {
    logger?.("before prepareRunIntegrationInputs", {
      ndays: inputLike.daily_lai.length,
      useJit,
    });
    const prepareStartedAt = performance.now();
    preparedInputs = prepareRunIntegrationInputs(inputLike);
    logger?.("after prepareRunIntegrationInputs", {
      elapsedMs: performance.now() - prepareStartedAt,
      hourlyShape: preparedInputs.hourly_temp.shape,
      laiShape: preparedInputs.daily_lai.shape,
      yassoParamShape: preparedInputs.yasso_param.shape,
    });

    const runner = useJit ? runIntegrationJitted : runIntegration;
    logger?.("selected runner", { mode: useJit ? "jit" : "eager" });
    const runStartedAt = performance.now();
    [finalCarry, dailyOutputs] = runner(preparedInputs);
    logger?.("after runner invocation", {
      elapsedMs: performance.now() - runStartedAt,
      outputKeys: Object.keys(dailyOutputs),
    });

    const daily = await extractDailyOutputs(dailyOutputs);
    logger?.("after output extraction", {
      gppLength: daily.gpp.length,
      neeLength: daily.nee.length,
      lastSoc: daily.socTotal.at(-1),
    });

    return {
      elapsedMs: performance.now() - runStartedAt,
      executionLabel: useJit ? "jit" : "eager",
      daily,
    };
  } finally {
    if (dailyOutputs) tree.dispose(dailyOutputs);
    if (finalCarry) tree.dispose(finalCarry);
    if (preparedInputs) tree.dispose(preparedInputs);
  }
}