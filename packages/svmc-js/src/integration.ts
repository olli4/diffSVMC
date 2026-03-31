import { lax, tree } from "@hamk-uas/jax-js-nonconsuming";
import { np } from "./precision.js";
import { allocHypothesis2Fn, invertAllocFn } from "./allocation/index.js";
import { densityH2o, pmodelHydraulicsNumerical } from "./phydro/index.js";
import {
  canopyWaterFlux,
  soilHydraulicConductivity,
  soilWater,
  soilWaterRetentionCurve,
  type CanopySnowParams,
  type CanopyWaterState,
  type SoilWaterState,
  type SpafhyAeroParams,
} from "./water/index.js";
import type { SoilHydroParams } from "./water/soil-hydraulics.js";
import { decomposeFn, initializeTotcFn, inputsToFractions } from "./yasso/index.js";

const C_MOLMASS = 12.0107;
const H2O_MOLMASS = 18.01528;
const KPHIO = 0.087182;
const K_EXT = 0.5;
const TIME_STEP = 1.0;
const ALPHA_SMOOTH1 = 0.01;
const ALPHA_SMOOTH2 = 0.0016;
const LAI_GUARD = 1.0e-6;

export interface RunIntegrationInputs {
  hourly_temp: np.ArrayLike;
  hourly_rg: np.ArrayLike;
  hourly_prec: np.ArrayLike;
  hourly_vpd: np.ArrayLike;
  hourly_pres: np.ArrayLike;
  hourly_co2: np.ArrayLike;
  hourly_wind: np.ArrayLike;
  daily_lai: np.ArrayLike;
  daily_manage_type: np.ArrayLike;
  daily_manage_c_in: np.ArrayLike;
  daily_manage_c_out: np.ArrayLike;
  conductivity: np.ArrayLike;
  psi50: np.ArrayLike;
  b_param: np.ArrayLike;
  alpha_cost: np.ArrayLike;
  gamma_cost: np.ArrayLike;
  rdark: np.ArrayLike;
  soil_depth: np.ArrayLike;
  max_poros: np.ArrayLike;
  fc: np.ArrayLike;
  wp: np.ArrayLike;
  ksat: np.ArrayLike;
  n_van: np.ArrayLike;
  watres: np.ArrayLike;
  alpha_van: np.ArrayLike;
  watsat: np.ArrayLike;
  maxpond: np.ArrayLike;
  wmax: np.ArrayLike;
  wmaxsnow: np.ArrayLike;
  kmelt: np.ArrayLike;
  kfreeze: np.ArrayLike;
  frac_snowliq: np.ArrayLike;
  gsoil: np.ArrayLike;
  hc: np.ArrayLike;
  w_leaf: np.ArrayLike;
  rw: np.ArrayLike;
  rwmin: np.ArrayLike;
  zmeas: np.ArrayLike;
  zground: np.ArrayLike;
  zo_ground: np.ArrayLike;
  cratio_resp: np.ArrayLike;
  cratio_leaf: np.ArrayLike;
  cratio_root: np.ArrayLike;
  cratio_biomass: np.ArrayLike;
  harvest_index: np.ArrayLike;
  turnover_cleaf: np.ArrayLike;
  turnover_croot: np.ArrayLike;
  sla: np.ArrayLike;
  q10: np.ArrayLike;
  invert_option: np.ArrayLike;
  pft_is_oat: np.ArrayLike;
  yasso_param: np.ArrayLike;
  yasso_totc: np.ArrayLike;
  yasso_cn_input: np.ArrayLike;
  yasso_fract_root: np.ArrayLike;
  yasso_fract_legacy: np.ArrayLike;
  yasso_tempr_c: np.ArrayLike;
  yasso_precip_day: np.ArrayLike;
  yasso_tempr_ampl: np.ArrayLike;
  phydro_optimizer?: string;
}

interface HourlyCarry {
  cw_state: CanopyWaterState;
  sw_state: SoilWaterState;
  met_rolling: np.Array;
  is_first_met: np.Array;
  temp_acc: np.Array;
  precip_acc: np.Array;
  gpp_acc: np.Array;
  vcmax_acc: np.Array;
  num_gpp: np.Array;
  num_vcmax: np.Array;
  et_acc: np.Array;
}

export interface DailyCarry {
  cw_state: CanopyWaterState;
  sw_state: SoilWaterState;
  met_rolling: np.Array;
  is_first_met: np.Array;
  cleaf: np.Array;
  croot: np.Array;
  cstem: np.Array;
  cgrain: np.Array;
  litter_cleaf: np.Array;
  litter_croot: np.Array;
  compost: np.Array;
  soluble: np.Array;
  above: np.Array;
  below: np.Array;
  yield_: np.Array;
  grain_fill: np.Array;
  lai_alloc: np.Array;
  pheno: np.Array;
  cstate: np.Array;
  nstate: np.Array;
  lai_prev: np.Array;
}

interface HourlyForcing {
  temp_k: np.Array;
  rg: np.Array;
  prec: np.Array;
  vpd: np.Array;
  pres: np.Array;
  co2: np.Array;
  wind: np.Array;
}

interface DailyForcing {
  hourly_temp: np.Array;
  hourly_rg: np.Array;
  hourly_prec: np.Array;
  hourly_vpd: np.Array;
  hourly_pres: np.Array;
  hourly_co2: np.Array;
  hourly_wind: np.Array;
  lai: np.Array;
  management_type: np.Array;
  management_c_in: np.Array;
  management_c_out: np.Array;
}

export interface DailyOutput {
  gpp_avg: np.Array;
  nee: np.Array;
  hetero_resp: np.Array;
  auto_resp: np.Array;
  cleaf: np.Array;
  croot: np.Array;
  cstem: np.Array;
  cgrain: np.Array;
  lai_alloc: np.Array;
  litter_cleaf: np.Array;
  litter_croot: np.Array;
  soc_total: np.Array;
  wliq: np.Array;
  psi: np.Array;
  cstate: np.Array;
  et_total: np.Array;
}

function leadingLength(value: np.ArrayLike): number {
  if (Array.isArray(value)) return value.length;
  const maybeShape = value as { shape?: number[] };
  if (Array.isArray(maybeShape.shape) && maybeShape.shape.length > 0) {
    return maybeShape.shape[0];
  }
  throw new TypeError("Unable to infer leading dimension for integration inputs");
}

function cloneArray(value: np.Array): np.Array {
  return value.ref;
}

function toJs<T>(value: np.ArrayLike): T {
  if (value != null && typeof value === "object" && "js" in value) {
    return (value as np.Array).js() as T;
  }
  return value as T;
}

function scalar(value: np.Array): number {
  return value.item() as number;
}

function disposeCanopyWaterState(state: CanopyWaterState): void {
  state.CanopyStorage.dispose();
  state.SWE.dispose();
  state.swe_i.dispose();
  state.swe_l.dispose();
}

function disposeSoilWaterState(state: SoilWaterState): void {
  state.WatSto.dispose();
  state.PondSto.dispose();
  state.MaxWatSto.dispose();
  state.MaxPondSto.dispose();
  state.FcSto.dispose();
  state.Wliq.dispose();
  state.Psi.dispose();
  state.Sat.dispose();
  state.Kh.dispose();
  state.beta.dispose();
}

function cloneCanopyWaterState(state: CanopyWaterState): CanopyWaterState {
  return {
    CanopyStorage: cloneArray(state.CanopyStorage),
    SWE: cloneArray(state.SWE),
    swe_i: cloneArray(state.swe_i),
    swe_l: cloneArray(state.swe_l),
  };
}

function cloneSoilWaterState(state: SoilWaterState): SoilWaterState {
  return {
    WatSto: cloneArray(state.WatSto),
    PondSto: cloneArray(state.PondSto),
    MaxWatSto: cloneArray(state.MaxWatSto),
    MaxPondSto: cloneArray(state.MaxPondSto),
    FcSto: cloneArray(state.FcSto),
    Wliq: cloneArray(state.Wliq),
    Psi: cloneArray(state.Psi),
    Sat: cloneArray(state.Sat),
    Kh: cloneArray(state.Kh),
    beta: cloneArray(state.beta),
  };
}

function cloneDailyCarry(carry: DailyCarry): DailyCarry {
  return {
    cw_state: cloneCanopyWaterState(carry.cw_state),
    sw_state: cloneSoilWaterState(carry.sw_state),
    met_rolling: cloneArray(carry.met_rolling),
    is_first_met: cloneArray(carry.is_first_met),
    cleaf: cloneArray(carry.cleaf),
    croot: cloneArray(carry.croot),
    cstem: cloneArray(carry.cstem),
    cgrain: cloneArray(carry.cgrain),
    litter_cleaf: cloneArray(carry.litter_cleaf),
    litter_croot: cloneArray(carry.litter_croot),
    compost: cloneArray(carry.compost),
    soluble: cloneArray(carry.soluble),
    above: cloneArray(carry.above),
    below: cloneArray(carry.below),
    yield_: cloneArray(carry.yield_),
    grain_fill: cloneArray(carry.grain_fill),
    lai_alloc: cloneArray(carry.lai_alloc),
    pheno: cloneArray(carry.pheno),
    cstate: cloneArray(carry.cstate),
    nstate: cloneArray(carry.nstate),
    lai_prev: cloneArray(carry.lai_prev),
  };
}

function cloneDailyOutput(output: DailyOutput): DailyOutput {
  return {
    gpp_avg: cloneArray(output.gpp_avg),
    nee: cloneArray(output.nee),
    hetero_resp: cloneArray(output.hetero_resp),
    auto_resp: cloneArray(output.auto_resp),
    cleaf: cloneArray(output.cleaf),
    croot: cloneArray(output.croot),
    cstem: cloneArray(output.cstem),
    cgrain: cloneArray(output.cgrain),
    lai_alloc: cloneArray(output.lai_alloc),
    litter_cleaf: cloneArray(output.litter_cleaf),
    litter_croot: cloneArray(output.litter_croot),
    soc_total: cloneArray(output.soc_total),
    wliq: cloneArray(output.wliq),
    psi: cloneArray(output.psi),
    cstate: cloneArray(output.cstate),
    et_total: cloneArray(output.et_total),
  };
}

export function initializationSpafhy(
  soil_depth: np.ArrayLike,
  max_poros: np.ArrayLike,
  fc: np.ArrayLike,
  maxpond: np.ArrayLike,
  soil_params: SoilHydroParams,
): [CanopyWaterState, SoilWaterState] {
  using soilDepth = np.array(soil_depth);
  using maxPoros = np.array(max_poros);
  using fcArr = np.array(fc);
  using maxpondArr = np.array(maxpond);

  using maxWatStoBase = soilDepth.mul(maxPoros);
  using maxWatSto = maxWatStoBase.mul(1000.0);
  using fcStoBase = soilDepth.mul(fcArr);
  using fcSto = fcStoBase.mul(1000.0);
  using watSto = maxWatSto.mul(0.9);
  const pondSto = np.array(0.0);
  using watStoRatio = watSto.div(maxWatSto);
  using satScale = np.minimum(1.0, watStoRatio);
  const wliq = maxPoros.mul(satScale);
  const sat = wliq.div(maxPoros);
  const beta = np.minimum(1.0, sat);
  const psi = soilWaterRetentionCurve(wliq, soil_params);
  const kh = soilHydraulicConductivity(wliq, soil_params);

  const cw_state: CanopyWaterState = {
    CanopyStorage: np.array(0.0),
    SWE: np.array(0.0),
    swe_i: np.array(0.0),
    swe_l: np.array(0.0),
  };
  const sw_state: SoilWaterState = {
    WatSto: watSto.ref,
    PondSto: pondSto,
    MaxWatSto: maxWatSto.ref,
    MaxPondSto: maxpondArr.ref,
    FcSto: fcSto.ref,
    Wliq: wliq.ref,
    Psi: psi,
    Sat: sat.ref,
    Kh: kh,
    beta: beta.ref,
  };
  return [cw_state, sw_state];
}

function makeHourlyStep(
  lai: np.Array,
  fapar: np.Array,
  aero_params: SpafhyAeroParams,
  cs_params: CanopySnowParams,
  soil_params: SoilHydroParams,
  max_poros: np.Array,
  rdark: np.Array,
  conductivity: np.Array,
  psi50: np.Array,
  b_param: np.Array,
  alpha_cost: np.Array,
  gamma_cost: np.Array,
  solverKind: "projected_lbfgs" | "projected_newton",
): (carry: HourlyCarry, forcing: HourlyForcing) => [HourlyCarry, null] {
  return (carry: HourlyCarry, forcing: HourlyForcing): [HourlyCarry, null] => {
    using time_step = np.array(TIME_STEP);
    using tc = forcing.temp_k.sub(273.15);
    using lai_safe = np.maximum(lai, LAI_GUARD);
    using ppfd_base = forcing.rg.mul(2.1);
    using ppfd = ppfd_base.div(lai_safe);
    using co2_ppm = forcing.co2.mul(1.0e6);

    const phydro = pmodelHydraulicsNumerical(
      tc,
      ppfd,
      forcing.vpd,
      co2_ppm,
      forcing.pres,
      fapar,
      carry.sw_state.Psi,
      rdark,
      conductivity,
      psi50,
      b_param,
      alpha_cost,
      gamma_cost,
      KPHIO,
      solverKind,
    );
    using aj = phydro.aj;
    using gs = phydro.gs;
    using vcmax_hr = phydro.vcmax;
    phydro.jmax.dispose();
    phydro.dpsi.dispose();
    phydro.ci.dispose();
    phydro.chi.dispose();
    phydro.profit.dispose();
    phydro.chiJmaxLim.dispose();

    using gpp_inner = rdark.mul(vcmax_hr);
    using gpp_sum = aj.add(gpp_inner);
    using gpp_scaled0 = gpp_sum.mul(C_MOLMASS);
    using gpp_scaled1 = gpp_scaled0.mul(1.0e-6);
    using gpp_scaled2 = gpp_scaled1.mul(1.0e-3);
    using gpp_scaled = gpp_scaled2.mul(lai);
    using lai_positive = lai.greater(LAI_GUARD);
    using rg_positive = forcing.rg.greater(0.0);
    using has_light = lai_positive.mul(rg_positive);
    using gpp_hr = np.where(has_light, gpp_scaled, 0.0);
    using aj_guarded = np.where(has_light, aj, 0.0);
    using gs_guarded = np.where(has_light, gs, 0.0);
    using vcmax_guarded = np.where(has_light, vcmax_hr, 0.0);

    using rho_w = densityH2o(tc, forcing.pres);
    using vpd_over_pres = forcing.vpd.div(forcing.pres);
    using tr_scaled0 = gs_guarded.mul(1.6);
    using tr_scaled1 = tr_scaled0.mul(vpd_over_pres);
    using tr_scaled = tr_scaled1.mul(H2O_MOLMASS);
    using tr_raw_base = tr_scaled.div(rho_w);
    using tr_raw = tr_raw_base.mul(lai);
    using gs_is_finite = np.isfinite(gs_guarded);
    using tr_valid = gs_is_finite.mul(has_light);
    using tr_phydro = np.where(tr_valid, tr_raw, 0.0);

    using rn = forcing.rg.mul(0.7);
    const [cw_state, cw_flux] = canopyWaterFlux(
      rn,
      tc,
      forcing.prec,
      forcing.vpd,
      forcing.wind,
      forcing.pres,
      fapar,
      lai,
      carry.cw_state,
      carry.sw_state.beta,
      carry.sw_state.WatSto,
      aero_params,
      cs_params,
      time_step,
    );

    using tr_spafhy = tr_phydro.mul(TIME_STEP * 3600.0);
    using et_from_canopy = cw_flux.SoilEvap.add(cw_flux.CanopyEvap);
    using et_hr = et_from_canopy.add(tr_spafhy);

    const soil_result = soilWater(
      carry.sw_state,
      soil_params,
      max_poros,
      cw_flux.PotInfiltration,
      tr_spafhy,
      cw_flux.SoilEvap,
      np.array(0.0),
      time_step,
    );

    using met_prec_melt = cw_flux.Melt.div(TIME_STEP * 3600.0);
    using met_prec = forcing.prec.add(met_prec_melt);
    using met_daily = np.stack([tc, met_prec]);
    using smoothed_temp_old = carry.met_rolling.slice(0);
    using smoothed_prec_old = carry.met_rolling.slice(1);
    using met_daily_temp = met_daily.slice(0);
    using met_daily_prec = met_daily.slice(1);
    using smooth_temp_new = met_daily_temp.mul(ALPHA_SMOOTH1);
    using smooth_temp_old_scaled = smoothed_temp_old.mul(1.0 - ALPHA_SMOOTH1);
    using smooth_temp = smooth_temp_new.add(smooth_temp_old_scaled);
    using smooth_prec_new = met_daily_prec.mul(ALPHA_SMOOTH2);
    using smooth_prec_old_scaled = smoothed_prec_old.mul(1.0 - ALPHA_SMOOTH2);
    using smooth_prec = smooth_prec_new.add(smooth_prec_old_scaled);
    using smoothed = np.stack([smooth_temp, smooth_prec]);
    const new_met_rolling = np.where(carry.is_first_met, met_daily, smoothed);

    using gpp_valid_finite = np.isfinite(aj_guarded);
    using gpp_valid_nonneg = aj_guarded.greaterEqual(0.0);
    using gpp_valid = gpp_valid_finite.mul(gpp_valid_nonneg);
    using gpp_to_add = np.where(gpp_valid, gpp_hr, 0.0);
    using gpp_count_inc = np.where(gpp_valid, 1.0, 0.0);
    const num_gpp = carry.num_gpp.add(gpp_count_inc);

    using vcmax_valid_finite = np.isfinite(vcmax_guarded);
    using vcmax_valid_positive = vcmax_guarded.greater(0.0);
    using vcmax_valid = vcmax_valid_finite.mul(vcmax_valid_positive);
    using vcmax_to_add = np.where(vcmax_valid, vcmax_guarded, 0.0);
    using vcmax_count_inc = np.where(vcmax_valid, 1.0, 0.0);
    const num_vcmax = carry.num_vcmax.add(vcmax_count_inc);

    using new_met_temp = new_met_rolling.slice(0);
    using new_met_prec = new_met_rolling.slice(1);
    using new_temp_acc = carry.temp_acc.add(new_met_temp);
    using precip_increment = new_met_prec.mul(TIME_STEP * 3600.0);
    using new_precip_acc = carry.precip_acc.add(precip_increment);
    using new_gpp_acc = carry.gpp_acc.add(gpp_to_add);
    using new_vcmax_acc = carry.vcmax_acc.add(vcmax_to_add);
    using new_et_acc = carry.et_acc.add(et_hr);
    const next_sw_state: SoilWaterState = {
      WatSto: soil_result.state.WatSto.ref,
      PondSto: soil_result.state.PondSto.ref,
      MaxWatSto: soil_result.state.MaxWatSto.ref,
      MaxPondSto: soil_result.state.MaxPondSto.ref,
      FcSto: soil_result.state.FcSto.ref,
      Wliq: soil_result.state.Wliq.ref,
      Psi: soil_result.state.Psi.ref,
      Sat: soil_result.state.Sat.ref,
      Kh: soil_result.state.Kh.ref,
      beta: soil_result.state.beta.ref,
    };
    const next_carry: HourlyCarry = {
      cw_state,
      sw_state: next_sw_state,
      met_rolling: new_met_rolling,
      is_first_met: np.array(false),
      temp_acc: new_temp_acc.ref,
      precip_acc: new_precip_acc.ref,
      gpp_acc: new_gpp_acc.ref,
      vcmax_acc: new_vcmax_acc.ref,
      num_gpp,
      num_vcmax,
      et_acc: new_et_acc.ref,
    };

    cw_flux.Throughfall.dispose();
    cw_flux.Interception.dispose();
    cw_flux.CanopyEvap.dispose();
    cw_flux.Unloading.dispose();
    cw_flux.SoilEvap.dispose();
    cw_flux.ET.dispose();
    cw_flux.Transpiration.dispose();
    cw_flux.PotInfiltration.dispose();
    cw_flux.Melt.dispose();
    cw_flux.Freeze.dispose();
    cw_flux.mbe.dispose();
    soil_result.flux.Infiltration.dispose();
    soil_result.flux.Runoff.dispose();
    soil_result.flux.Drainage.dispose();
    soil_result.flux.LateralFlow.dispose();
    soil_result.flux.ET.dispose();
    soil_result.flux.mbe.dispose();
    soil_result.state.WatSto.dispose();
    soil_result.state.PondSto.dispose();
    soil_result.state.MaxWatSto.dispose();
    soil_result.state.MaxPondSto.dispose();
    soil_result.state.FcSto.dispose();
    soil_result.state.Wliq.dispose();
    soil_result.state.Psi.dispose();
    soil_result.state.Sat.dispose();
    soil_result.state.Kh.dispose();
    soil_result.state.beta.dispose();
    soil_result.trOut.dispose();
    soil_result.evapOut.dispose();
    soil_result.latflowOut.dispose();

    return [next_carry, null];
  };
}

function makeDailyStep(
  aero_params: SpafhyAeroParams,
  cs_params: CanopySnowParams,
  soil_params: SoilHydroParams,
  max_poros: np.Array,
  rdark: np.Array,
  conductivity: np.Array,
  psi50: np.Array,
  b_param: np.Array,
  alpha_cost: np.Array,
  gamma_cost: np.Array,
  cratio_resp: np.Array,
  cratio_leaf: np.Array,
  cratio_root: np.Array,
  cratio_biomass: np.Array,
  harvest_index: np.Array,
  turnover_cleaf: np.Array,
  turnover_croot: np.Array,
  sla: np.Array,
  q10: np.Array,
  invert_option: np.Array,
  yasso_param: np.Array,
  pft_is_oat: np.Array,
  solverKind: "projected_lbfgs" | "projected_newton",
): (carry: DailyCarry, forcing: DailyForcing) => [DailyCarry, DailyOutput] {
  return (carry: DailyCarry, forcing: DailyForcing): [DailyCarry, DailyOutput] => {
    using delta_lai = forcing.lai.sub(carry.lai_prev);
    using fapar_arg = forcing.lai.mul(-K_EXT);
    using fapar_exp = np.exp(fapar_arg);
    using one = np.array(1.0);
    using fapar = one.sub(fapar_exp);

    const hourly_step = makeHourlyStep(
      forcing.lai,
      fapar,
      aero_params,
      cs_params,
      soil_params,
      max_poros,
      rdark,
      conductivity,
      psi50,
      b_param,
      alpha_cost,
      gamma_cost,
      solverKind,
    );

    const hourly_forcing: HourlyForcing = {
      temp_k: forcing.hourly_temp,
      rg: forcing.hourly_rg,
      prec: forcing.hourly_prec,
      vpd: forcing.hourly_vpd,
      pres: forcing.hourly_pres,
      co2: forcing.hourly_co2,
      wind: forcing.hourly_wind,
    };

    const hourly_init: HourlyCarry = {
      cw_state: carry.cw_state,
      sw_state: carry.sw_state,
      met_rolling: carry.met_rolling,
      is_first_met: carry.is_first_met,
      temp_acc: np.array(0.0),
      precip_acc: np.array(0.0),
      gpp_acc: np.array(0.0),
      vcmax_acc: np.array(0.0),
      num_gpp: np.array(0.0),
      num_vcmax: np.array(0.0),
      et_acc: np.array(0.0),
    };

    const hourly_body = (hourIdx: np.Array, hourlyCarry: HourlyCarry): HourlyCarry => {
      using temp_k_slice = lax.dynamicSlice(hourly_forcing.temp_k, [hourIdx], [1]);
      using rg_slice = lax.dynamicSlice(hourly_forcing.rg, [hourIdx], [1]);
      using prec_slice = lax.dynamicSlice(hourly_forcing.prec, [hourIdx], [1]);
      using vpd_slice = lax.dynamicSlice(hourly_forcing.vpd, [hourIdx], [1]);
      using pres_slice = lax.dynamicSlice(hourly_forcing.pres, [hourIdx], [1]);
      using co2_slice = lax.dynamicSlice(hourly_forcing.co2, [hourIdx], [1]);
      using wind_slice = lax.dynamicSlice(hourly_forcing.wind, [hourIdx], [1]);
      using temp_k = temp_k_slice.reshape([]);
      using rg = rg_slice.reshape([]);
      using prec = prec_slice.reshape([]);
      using vpd = vpd_slice.reshape([]);
      using pres = pres_slice.reshape([]);
      using co2 = co2_slice.reshape([]);
      using wind = wind_slice.reshape([]);
      const [nextCarry] = hourly_step(hourlyCarry, {
        temp_k,
        rg,
        prec,
        vpd,
        pres,
        co2,
        wind,
      });
      return nextCarry;
    };

    const final_hourly = lax.foriLoop(0, 24, hourly_body, hourly_init);

    using temp_avg = final_hourly.temp_acc.div(24.0);
    const gpp_avg = np.where(
      final_hourly.num_gpp.greater(0.0),
      final_hourly.gpp_acc.div(final_hourly.num_gpp),
      0.0,
    );
    using vcmax_avg = np.where(
      final_hourly.num_vcmax.greater(0.0),
      final_hourly.vcmax_acc.div(final_hourly.num_vcmax),
      0.0,
    );
    const precip_acc = final_hourly.precip_acc;
    using leaf_rdark_base = rdark.mul(vcmax_avg);
    using leaf_rdark_scale0 = leaf_rdark_base.mul(C_MOLMASS);
    using leaf_rdark_scale1 = leaf_rdark_scale0.mul(1.0e-6);
    using leaf_rdark_scale = leaf_rdark_scale1.mul(1.0e-3);
    using leaf_rdark_day = leaf_rdark_scale.mul(forcing.lai);

    const inv_result = invertAllocFn(
      delta_lai,
      leaf_rdark_day,
      temp_avg,
      gpp_avg,
      carry.litter_cleaf,
      carry.cleaf,
      carry.cstem,
      cratio_resp,
      cratio_leaf,
      cratio_root,
      cratio_biomass,
      harvest_index,
      turnover_cleaf,
      turnover_croot,
      sla,
      q10,
      invert_option,
      forcing.management_type,
      forcing.management_c_in,
      forcing.management_c_out,
      pft_is_oat,
      carry.pheno,
    );
    inv_result.deltaLai.dispose();

    const alloc_result = allocHypothesis2Fn(
      temp_avg,
      gpp_avg,
      leaf_rdark_day,
      carry.croot,
      inv_result.cleaf,
      carry.cstem,
      carry.cgrain,
      inv_result.litterCleaf,
      carry.grain_fill,
      cratio_resp,
      inv_result.cratioLeaf,
      inv_result.cratioRoot,
      cratio_biomass,
      inv_result.turnoverCleaf,
      turnover_croot,
      sla,
      q10,
      invert_option,
      forcing.management_type,
      forcing.management_c_in,
      forcing.management_c_out,
      pft_is_oat,
      carry.pheno,
    );
    inv_result.litterCleaf.dispose();
    inv_result.cleaf.dispose();
    inv_result.cratioLeaf.dispose();
    inv_result.cratioRoot.dispose();
    inv_result.turnoverCleaf.dispose();

    const input_cfract = inputsToFractions(
      alloc_result.litterCleaf,
      alloc_result.litterCroot,
      carry.soluble,
      alloc_result.compost,
    );
    using one_day = np.array(1.0);
    const decomp_result = decomposeFn(
      yasso_param,
      one_day,
      temp_avg,
      precip_acc,
      carry.cstate,
      carry.nstate,
    );
    using cstate_plus_ctend = carry.cstate.add(decomp_result.ctend);
    const new_cstate = cstate_plus_ctend.add(input_cfract);
    const new_nstate = carry.nstate.add(decomp_result.ntend);

    using ctend_neg = decomp_result.ctend.neg();
    using hetero_resp_num = np.sum(ctend_neg);
    using hetero_resp_day = hetero_resp_num.div(24.0);
    const hetero_resp = hetero_resp_day.div(3600.0);
    using auto_resp_day = alloc_result.autoResp.div(24.0);
    const auto_resp = auto_resp_day.div(3600.0);
    using total_resp = hetero_resp.add(auto_resp);
    const nee = total_resp.sub(gpp_avg);
    const soc_total = np.sum(new_cstate);

    const next_carry: DailyCarry = {
      cw_state: {
        CanopyStorage: final_hourly.cw_state.CanopyStorage.ref,
        SWE: final_hourly.cw_state.SWE.ref,
        swe_i: final_hourly.cw_state.swe_i.ref,
        swe_l: final_hourly.cw_state.swe_l.ref,
      },
      sw_state: {
        WatSto: final_hourly.sw_state.WatSto.ref,
        PondSto: final_hourly.sw_state.PondSto.ref,
        MaxWatSto: final_hourly.sw_state.MaxWatSto.ref,
        MaxPondSto: final_hourly.sw_state.MaxPondSto.ref,
        FcSto: final_hourly.sw_state.FcSto.ref,
        Wliq: final_hourly.sw_state.Wliq.ref,
        Psi: final_hourly.sw_state.Psi.ref,
        Sat: final_hourly.sw_state.Sat.ref,
        Kh: final_hourly.sw_state.Kh.ref,
        beta: final_hourly.sw_state.beta.ref,
      },
      met_rolling: final_hourly.met_rolling.ref,
      is_first_met: final_hourly.is_first_met.ref,
      cleaf: alloc_result.cleaf.ref,
      croot: alloc_result.croot.ref,
      cstem: alloc_result.cstem.ref,
      cgrain: alloc_result.cgrain.ref,
      litter_cleaf: alloc_result.litterCleaf.ref,
      litter_croot: alloc_result.litterCroot.ref,
      compost: alloc_result.compost.ref,
      soluble: carry.soluble.ref,
      above: alloc_result.abovebiomass.ref,
      below: alloc_result.belowbiomass.ref,
      yield_: alloc_result.yield.ref,
      grain_fill: alloc_result.grainFill.ref,
      lai_alloc: alloc_result.lai.ref,
      pheno: alloc_result.phenoStage.ref,
      cstate: new_cstate.ref,
      nstate: new_nstate.ref,
      lai_prev: forcing.lai.ref,
    };

    const output: DailyOutput = {
      gpp_avg: gpp_avg.ref,
      nee: nee.ref,
      hetero_resp: hetero_resp.ref,
      auto_resp: auto_resp.ref,
      cleaf: alloc_result.cleaf.ref,
      croot: alloc_result.croot.ref,
      cstem: alloc_result.cstem.ref,
      cgrain: alloc_result.cgrain.ref,
      lai_alloc: alloc_result.lai.ref,
      litter_cleaf: alloc_result.litterCleaf.ref,
      litter_croot: alloc_result.litterCroot.ref,
      soc_total: soc_total.ref,
      wliq: final_hourly.sw_state.Wliq.ref,
      psi: final_hourly.sw_state.Psi.ref,
      cstate: new_cstate.ref,
      et_total: final_hourly.et_acc.ref,
    };

    alloc_result.nppDay.dispose();
    alloc_result.autoResp.dispose();
    alloc_result.croot.dispose();
    alloc_result.cleaf.dispose();
    alloc_result.cstem.dispose();
    alloc_result.cgrain.dispose();
    alloc_result.litterCleaf.dispose();
    alloc_result.litterCroot.dispose();
    alloc_result.compost.dispose();
    alloc_result.lai.dispose();
    alloc_result.abovebiomass.dispose();
    alloc_result.belowbiomass.dispose();
    alloc_result.yield.dispose();
    alloc_result.grainFill.dispose();
    alloc_result.phenoStage.dispose();
    gpp_avg.dispose();
    hetero_resp.dispose();
    auto_resp.dispose();
    nee.dispose();
    soc_total.dispose();
    new_cstate.dispose();
    new_nstate.dispose();
    decomp_result.ctend.dispose();
    decomp_result.ntend.dispose();
    input_cfract.dispose();
    tree.dispose(final_hourly);

    return [next_carry, output];
  };
}

export function runIntegration(inputs: RunIntegrationInputs): [DailyCarry, DailyOutput] {
  if (
    inputs.phydro_optimizer != null
    && inputs.phydro_optimizer !== "projected_lbfgs"
    && inputs.phydro_optimizer !== "projected_newton"
  ) {
    throw new Error(
      `Unknown phydro optimizer ${inputs.phydro_optimizer}; expected projected_lbfgs or projected_newton`,
    );
  }
  const solverKind = inputs.phydro_optimizer ?? "projected_lbfgs";

  const ndays = leadingLength(inputs.daily_lai);
  const hourly_temp = toJs<number[][]>(inputs.hourly_temp);
  const hourly_rg = toJs<number[][]>(inputs.hourly_rg);
  const hourly_prec = toJs<number[][]>(inputs.hourly_prec);
  const hourly_vpd = toJs<number[][]>(inputs.hourly_vpd);
  const hourly_pres = toJs<number[][]>(inputs.hourly_pres);
  const hourly_co2 = toJs<number[][]>(inputs.hourly_co2);
  const hourly_wind = toJs<number[][]>(inputs.hourly_wind);
  const daily_lai = toJs<number[]>(inputs.daily_lai);
  const daily_manage_type = toJs<number[]>(inputs.daily_manage_type);
  const daily_manage_c_in = toJs<number[]>(inputs.daily_manage_c_in);
  const daily_manage_c_out = toJs<number[]>(inputs.daily_manage_c_out);

  const conductivity_val = toJs<number>(inputs.conductivity);
  const psi50_val = toJs<number>(inputs.psi50);
  const b_param_val = toJs<number>(inputs.b_param);
  const alpha_cost_val = toJs<number>(inputs.alpha_cost);
  const gamma_cost_val = toJs<number>(inputs.gamma_cost);
  const rdark_val = toJs<number>(inputs.rdark);
  const soil_depth_val = toJs<number>(inputs.soil_depth);
  const max_poros_val = toJs<number>(inputs.max_poros);
  const fc_val = toJs<number>(inputs.fc);
  const maxpond_val = toJs<number>(inputs.maxpond);

  const soil_params = tree.makeDisposable({
    watsat: np.array(inputs.watsat),
    watres: np.array(inputs.watres),
    alphaVan: np.array(inputs.alpha_van),
    nVan: np.array(inputs.n_van),
    ksat: np.array(inputs.ksat),
  }) as SoilHydroParams;
  const aero_params = tree.makeDisposable({
    hc: np.array(inputs.hc),
    zmeas: np.array(inputs.zmeas),
    zground: np.array(inputs.zground),
    zo_ground: np.array(inputs.zo_ground),
    w_leaf: np.array(inputs.w_leaf),
  }) as SpafhyAeroParams;
  const cs_params = tree.makeDisposable({
    wmax: np.array(inputs.wmax),
    wmaxsnow: np.array(inputs.wmaxsnow),
    kmelt: np.array(inputs.kmelt),
    kfreeze: np.array(inputs.kfreeze),
    fracSnowliq: np.array(inputs.frac_snowliq),
    gsoil: np.array(inputs.gsoil),
  }) as CanopySnowParams;
  const yasso_param = np.array(inputs.yasso_param);
  const allocation_params = tree.makeDisposable({
    cratio_resp: np.array(inputs.cratio_resp),
    cratio_leaf: np.array(inputs.cratio_leaf),
    cratio_root: np.array(inputs.cratio_root),
    cratio_biomass: np.array(inputs.cratio_biomass),
    harvest_index: np.array(inputs.harvest_index),
    turnover_cleaf: np.array(inputs.turnover_cleaf),
    turnover_croot: np.array(inputs.turnover_croot),
    sla: np.array(inputs.sla),
    q10: np.array(inputs.q10),
    invert_option: np.array(inputs.invert_option),
    pft_is_oat: np.array(inputs.pft_is_oat),
  }) as {
    cratio_resp: np.Array;
    cratio_leaf: np.Array;
    cratio_root: np.Array;
    cratio_biomass: np.Array;
    harvest_index: np.Array;
    turnover_cleaf: np.Array;
    turnover_croot: np.Array;
    sla: np.Array;
    q10: np.Array;
    invert_option: np.Array;
    pft_is_oat: np.Array;
  };

  try {
    const [cw_init, sw_init] = initializationSpafhy(
      soil_depth_val,
      max_poros_val,
      fc_val,
      maxpond_val,
      soil_params,
    );

    let cw_state = {
      CanopyStorage: scalar(cw_init.CanopyStorage),
      SWE: scalar(cw_init.SWE),
      swe_i: scalar(cw_init.swe_i),
      swe_l: scalar(cw_init.swe_l),
    };
    let sw_state = {
      WatSto: scalar(sw_init.WatSto),
      PondSto: scalar(sw_init.PondSto),
      MaxWatSto: scalar(sw_init.MaxWatSto),
      MaxPondSto: scalar(sw_init.MaxPondSto),
      FcSto: scalar(sw_init.FcSto),
      Wliq: scalar(sw_init.Wliq),
      Psi: scalar(sw_init.Psi),
      Sat: scalar(sw_init.Sat),
      Kh: scalar(sw_init.Kh),
      beta: scalar(sw_init.beta),
    };
    disposeCanopyWaterState(cw_init);
    disposeSoilWaterState(sw_init);

    using yasso_totc = np.array(inputs.yasso_totc);
    using yasso_cn_input = np.array(inputs.yasso_cn_input);
    using yasso_fract_root = np.array(inputs.yasso_fract_root);
    using yasso_fract_legacy = np.array(inputs.yasso_fract_legacy);
    using yasso_tempr_c = np.array(inputs.yasso_tempr_c);
    using yasso_precip_day = np.array(inputs.yasso_precip_day);
    using yasso_tempr_ampl = np.array(inputs.yasso_tempr_ampl);
    const yasso_init = initializeTotcFn(
      yasso_param,
      yasso_totc,
      yasso_cn_input,
      yasso_fract_root,
      yasso_fract_legacy,
      yasso_tempr_c,
      yasso_precip_day,
      yasso_tempr_ampl,
    );
    let cstate = yasso_init.cstate.js() as number[];
    let nstate = scalar(yasso_init.nstate);
    yasso_init.cstate.dispose();
    yasso_init.nstate.dispose();

    let met_rolling = [0.0, 0.0];
    let is_first_met = true;
    let cleaf = 0.0;
    let croot = 0.0;
    let cstem = 0.0;
    let cgrain = 0.0;
    let litter_cleaf = 0.0;
    let litter_croot = 0.0;
    let compost = 0.0;
    const soluble = 0.0;
    let above = 0.0;
    let below = 0.0;
    let yield_ = 0.0;
    let grain_fill = 0.0;
    let lai_alloc = 0.0;
    let pheno = 1.0;
    let lai_prev = 0.0;

    const outputs = {
      gpp_avg: [] as number[],
      nee: [] as number[],
      hetero_resp: [] as number[],
      auto_resp: [] as number[],
      cleaf: [] as number[],
      croot: [] as number[],
      cstem: [] as number[],
      cgrain: [] as number[],
      lai_alloc: [] as number[],
      litter_cleaf: [] as number[],
      litter_croot: [] as number[],
      soc_total: [] as number[],
      wliq: [] as number[],
      psi: [] as number[],
      cstate: [] as number[][],
      et_total: [] as number[],
    };

    for (let day_idx = 0; day_idx < ndays; day_idx += 1) {
      const lai = daily_lai[day_idx];
      const fapar = 1.0 - Math.exp(-K_EXT * lai);
      let temp_acc = 0.0;
      let precip_acc = 0.0;
      let gpp_acc = 0.0;
      let vcmax_acc = 0.0;
      let num_gpp = 0.0;
      let num_vcmax = 0.0;
      let et_acc = 0.0;

      for (let hour_idx = 0; hour_idx < 24; hour_idx += 1) {
        const temp_k = hourly_temp[day_idx][hour_idx];
        const rg = hourly_rg[day_idx][hour_idx];
        const prec = hourly_prec[day_idx][hour_idx];
        const vpd = hourly_vpd[day_idx][hour_idx];
        const pres = hourly_pres[day_idx][hour_idx];
        const co2 = hourly_co2[day_idx][hour_idx];
        const wind = hourly_wind[day_idx][hour_idx];
        const tc = temp_k - 273.15;
        const ppfd = rg * 2.1 / Math.max(lai, LAI_GUARD);
        const co2_ppm = co2 * 1.0e6;
        const has_light = lai > LAI_GUARD && rg > 0.0;

        let aj = 0.0;
        let gs = 0.0;
        let vcmax_hr = 0.0;
        if (has_light) {
          const phydro = pmodelHydraulicsNumerical(
            tc,
            ppfd,
            vpd,
            co2_ppm,
            pres,
            fapar,
            sw_state.Psi,
            rdark_val,
            conductivity_val,
            psi50_val,
            b_param_val,
            alpha_cost_val,
            gamma_cost_val,
            KPHIO,
            solverKind,
          );
          aj = scalar(phydro.aj);
          gs = scalar(phydro.gs);
          vcmax_hr = scalar(phydro.vcmax);
          phydro.jmax.dispose();
          phydro.dpsi.dispose();
          phydro.gs.dispose();
          phydro.aj.dispose();
          phydro.ci.dispose();
          phydro.chi.dispose();
          phydro.vcmax.dispose();
          phydro.profit.dispose();
          phydro.chiJmaxLim.dispose();
        }

        const gpp_hr = (aj + rdark_val * vcmax_hr) * C_MOLMASS * 1.0e-6 * 1.0e-3 * lai;

        let tr_phydro = 0.0;
        if (has_light && Number.isFinite(gs)) {
          let rho_w = 1.0;
          {
            using tc_arr = np.array(tc);
            using pres_arr = np.array(pres);
            using rho_w_arr = densityH2o(tc_arr, pres_arr);
            rho_w = scalar(rho_w_arr);
          }
          tr_phydro = 1.6 * gs * (vpd / pres) * H2O_MOLMASS / rho_w * lai;
        }
        const tr_spafhy = tr_phydro * TIME_STEP * 3600.0;

        {
          const cw_input: CanopyWaterState = {
            CanopyStorage: np.array(cw_state.CanopyStorage),
            SWE: np.array(cw_state.SWE),
            swe_i: np.array(cw_state.swe_i),
            swe_l: np.array(cw_state.swe_l),
          };
          const sw_input: SoilWaterState = {
            WatSto: np.array(sw_state.WatSto),
            PondSto: np.array(sw_state.PondSto),
            MaxWatSto: np.array(sw_state.MaxWatSto),
            MaxPondSto: np.array(sw_state.MaxPondSto),
            FcSto: np.array(sw_state.FcSto),
            Wliq: np.array(sw_state.Wliq),
            Psi: np.array(sw_state.Psi),
            Sat: np.array(sw_state.Sat),
            Kh: np.array(sw_state.Kh),
            beta: np.array(sw_state.beta),
          };
          using rn = np.array(rg * 0.7);
          using tc_arr = np.array(tc);
          using prec_arr = np.array(prec);
          using vpd_arr = np.array(vpd);
          using wind_arr = np.array(wind);
          using pres_arr = np.array(pres);
          using fapar_arr = np.array(fapar);
          using lai_arr = np.array(lai);
          using time_step = np.array(TIME_STEP);
          const [cw_output, cw_flux] = canopyWaterFlux(
            rn,
            tc_arr,
            prec_arr,
            vpd_arr,
            wind_arr,
            pres_arr,
            fapar_arr,
            lai_arr,
            cw_input,
            sw_input.beta,
            sw_input.WatSto,
            aero_params,
            cs_params,
            time_step,
          );
          using potinf = cw_flux.PotInfiltration.ref;
          using soil_evap = cw_flux.SoilEvap.ref;
          using latflow = np.array(0.0);
          using tr_arr = np.array(tr_spafhy);
          using max_poros_arr = np.array(max_poros_val);
          const soil_result = soilWater(
            sw_input,
            soil_params,
            max_poros_arr,
            potinf,
            tr_arr,
            soil_evap,
            latflow,
            time_step,
          );

          cw_state = {
            CanopyStorage: scalar(cw_output.CanopyStorage),
            SWE: scalar(cw_output.SWE),
            swe_i: scalar(cw_output.swe_i),
            swe_l: scalar(cw_output.swe_l),
          };
          sw_state = {
            WatSto: scalar(soil_result.state.WatSto),
            PondSto: scalar(soil_result.state.PondSto),
            MaxWatSto: scalar(soil_result.state.MaxWatSto),
            MaxPondSto: scalar(soil_result.state.MaxPondSto),
            FcSto: scalar(soil_result.state.FcSto),
            Wliq: scalar(soil_result.state.Wliq),
            Psi: scalar(soil_result.state.Psi),
            Sat: scalar(soil_result.state.Sat),
            Kh: scalar(soil_result.state.Kh),
            beta: scalar(soil_result.state.beta),
          };

          et_acc += tr_spafhy + scalar(cw_flux.SoilEvap) + scalar(cw_flux.CanopyEvap);
          const met_prec = prec + scalar(cw_flux.Melt) / (TIME_STEP * 3600.0);
          if (is_first_met) {
            met_rolling = [tc, met_prec];
            is_first_met = false;
          } else {
            met_rolling = [
              ALPHA_SMOOTH1 * tc + (1.0 - ALPHA_SMOOTH1) * met_rolling[0],
              ALPHA_SMOOTH2 * met_prec + (1.0 - ALPHA_SMOOTH2) * met_rolling[1],
            ];
          }

          temp_acc += met_rolling[0];
          precip_acc += met_rolling[1] * TIME_STEP * 3600.0;
          if (Number.isFinite(aj) && aj >= 0.0) {
            gpp_acc += gpp_hr;
            num_gpp += 1.0;
          }
          if (Number.isFinite(vcmax_hr) && vcmax_hr > 0.0) {
            vcmax_acc += vcmax_hr;
            num_vcmax += 1.0;
          }

          disposeCanopyWaterState(cw_input);
          disposeSoilWaterState(sw_input);
          disposeCanopyWaterState(cw_output);
          cw_flux.Throughfall.dispose();
          cw_flux.Interception.dispose();
          cw_flux.CanopyEvap.dispose();
          cw_flux.Unloading.dispose();
          cw_flux.SoilEvap.dispose();
          cw_flux.ET.dispose();
          cw_flux.Transpiration.dispose();
          cw_flux.PotInfiltration.dispose();
          cw_flux.Melt.dispose();
          cw_flux.Freeze.dispose();
          cw_flux.mbe.dispose();
          soil_result.state.WatSto.dispose();
          soil_result.state.PondSto.dispose();
          soil_result.state.Wliq.dispose();
          soil_result.state.Psi.dispose();
          soil_result.state.Sat.dispose();
          soil_result.state.Kh.dispose();
          soil_result.state.beta.dispose();
          soil_result.flux.Infiltration.dispose();
          soil_result.flux.Runoff.dispose();
          soil_result.flux.Drainage.dispose();
          soil_result.flux.LateralFlow.dispose();
          soil_result.flux.ET.dispose();
          soil_result.flux.mbe.dispose();
          soil_result.trOut.dispose();
          soil_result.evapOut.dispose();
          soil_result.latflowOut.dispose();
        }
      }

      const temp_avg = temp_acc / 24.0;
      const gpp_avg = num_gpp > 0.0 ? gpp_acc / num_gpp : 0.0;
      const vcmax_avg = num_vcmax > 0.0 ? vcmax_acc / num_vcmax : 0.0;
      const leaf_rdark_day = rdark_val * vcmax_avg * C_MOLMASS * 1.0e-6 * 1.0e-3 * lai;
      const delta_lai = lai - lai_prev;

      let inv_values: {
        litter_cleaf: number;
        cleaf: number;
        cratio_leaf: number;
        cratio_root: number;
        turnover_cleaf: number;
      };
      {
        using delta_lai_arr = np.array(delta_lai);
        using leaf_rdark_day_arr = np.array(leaf_rdark_day);
        using temp_avg_arr = np.array(temp_avg);
        using gpp_avg_arr = np.array(gpp_avg);
        using litter_cleaf_arr = np.array(litter_cleaf);
        using cleaf_arr = np.array(cleaf);
        using cstem_arr = np.array(cstem);
        using manage_type_arr = np.array(daily_manage_type[day_idx]);
        using manage_c_in_arr = np.array(daily_manage_c_in[day_idx]);
        using manage_c_out_arr = np.array(daily_manage_c_out[day_idx]);
        using pheno_arr = np.array(pheno);
        const inv_result = invertAllocFn(
          delta_lai_arr,
          leaf_rdark_day_arr,
          temp_avg_arr,
          gpp_avg_arr,
          litter_cleaf_arr,
          cleaf_arr,
          cstem_arr,
          allocation_params.cratio_resp,
          allocation_params.cratio_leaf,
          allocation_params.cratio_root,
          allocation_params.cratio_biomass,
          allocation_params.harvest_index,
          allocation_params.turnover_cleaf,
          allocation_params.turnover_croot,
          allocation_params.sla,
          allocation_params.q10,
          allocation_params.invert_option,
          manage_type_arr,
          manage_c_in_arr,
          manage_c_out_arr,
          allocation_params.pft_is_oat,
          pheno_arr,
        );
        inv_values = {
          litter_cleaf: scalar(inv_result.litterCleaf),
          cleaf: scalar(inv_result.cleaf),
          cratio_leaf: scalar(inv_result.cratioLeaf),
          cratio_root: scalar(inv_result.cratioRoot),
          turnover_cleaf: scalar(inv_result.turnoverCleaf),
        };
        inv_result.deltaLai.dispose();
        inv_result.litterCleaf.dispose();
        inv_result.cleaf.dispose();
        inv_result.cratioLeaf.dispose();
        inv_result.cratioRoot.dispose();
        inv_result.turnoverCleaf.dispose();
      }

      let alloc_values: {
        auto_resp: number;
        cleaf: number;
        croot: number;
        cstem: number;
        cgrain: number;
        litter_cleaf: number;
        litter_croot: number;
        compost: number;
        lai: number;
        above: number;
        below: number;
        yield_: number;
        grain_fill: number;
        pheno: number;
      };
      {
        using temp_avg_arr = np.array(temp_avg);
        using gpp_avg_arr = np.array(gpp_avg);
        using leaf_rdark_day_arr = np.array(leaf_rdark_day);
        using croot_arr = np.array(croot);
        using inv_cleaf_arr = np.array(inv_values.cleaf);
        using cstem_arr = np.array(cstem);
        using cgrain_arr = np.array(cgrain);
        using inv_litter_cleaf_arr = np.array(inv_values.litter_cleaf);
        using grain_fill_arr = np.array(grain_fill);
        using inv_cratio_leaf_arr = np.array(inv_values.cratio_leaf);
        using inv_cratio_root_arr = np.array(inv_values.cratio_root);
        using inv_turnover_cleaf_arr = np.array(inv_values.turnover_cleaf);
        using manage_type_arr = np.array(daily_manage_type[day_idx]);
        using manage_c_in_arr = np.array(daily_manage_c_in[day_idx]);
        using manage_c_out_arr = np.array(daily_manage_c_out[day_idx]);
        using pheno_arr = np.array(pheno);
        const alloc_result = allocHypothesis2Fn(
          temp_avg_arr,
          gpp_avg_arr,
          leaf_rdark_day_arr,
          croot_arr,
          inv_cleaf_arr,
          cstem_arr,
          cgrain_arr,
          inv_litter_cleaf_arr,
          grain_fill_arr,
          allocation_params.cratio_resp,
          inv_cratio_leaf_arr,
          inv_cratio_root_arr,
          allocation_params.cratio_biomass,
          inv_turnover_cleaf_arr,
          allocation_params.turnover_croot,
          allocation_params.sla,
          allocation_params.q10,
          allocation_params.invert_option,
          manage_type_arr,
          manage_c_in_arr,
          manage_c_out_arr,
          allocation_params.pft_is_oat,
          pheno_arr,
        );
        alloc_values = {
          auto_resp: scalar(alloc_result.autoResp),
          cleaf: scalar(alloc_result.cleaf),
          croot: scalar(alloc_result.croot),
          cstem: scalar(alloc_result.cstem),
          cgrain: scalar(alloc_result.cgrain),
          litter_cleaf: scalar(alloc_result.litterCleaf),
          litter_croot: scalar(alloc_result.litterCroot),
          compost: scalar(alloc_result.compost),
          lai: scalar(alloc_result.lai),
          above: scalar(alloc_result.abovebiomass),
          below: scalar(alloc_result.belowbiomass),
          yield_: scalar(alloc_result.yield),
          grain_fill: scalar(alloc_result.grainFill),
          pheno: scalar(alloc_result.phenoStage),
        };
        alloc_result.nppDay.dispose();
        alloc_result.autoResp.dispose();
        alloc_result.croot.dispose();
        alloc_result.cleaf.dispose();
        alloc_result.cstem.dispose();
        alloc_result.cgrain.dispose();
        alloc_result.litterCleaf.dispose();
        alloc_result.litterCroot.dispose();
        alloc_result.compost.dispose();
        alloc_result.lai.dispose();
        alloc_result.abovebiomass.dispose();
        alloc_result.belowbiomass.dispose();
        alloc_result.yield.dispose();
        alloc_result.grainFill.dispose();
        alloc_result.phenoStage.dispose();
      }

      let input_cfract: number[];
      {
        using litter_cleaf_arr = np.array(alloc_values.litter_cleaf);
        using litter_croot_arr = np.array(alloc_values.litter_croot);
        using soluble_arr = np.array(soluble);
        using compost_arr = np.array(alloc_values.compost);
        const input_cfract_arr = inputsToFractions(
          litter_cleaf_arr,
          litter_croot_arr,
          soluble_arr,
          compost_arr,
        );
        input_cfract = input_cfract_arr.js() as number[];
        input_cfract_arr.dispose();
      }

      let ctend: number[];
      let ntend = 0.0;
      {
        using one_day = np.array(1.0);
        using temp_avg_arr = np.array(temp_avg);
        using precip_acc_arr = np.array(precip_acc);
        using cstate_arr = np.array(cstate);
        using nstate_arr = np.array(nstate);
        const decomp_result = decomposeFn(
          yasso_param,
          one_day,
          temp_avg_arr,
          precip_acc_arr,
          cstate_arr,
          nstate_arr,
        );
        ctend = decomp_result.ctend.js() as number[];
        ntend = scalar(decomp_result.ntend);
        decomp_result.ctend.dispose();
        decomp_result.ntend.dispose();
      }

      cstate = cstate.map((value, index) => value + ctend[index] + input_cfract[index]);
      nstate += ntend;

      const hetero_resp = ctend.reduce((sum, value) => sum - value, 0.0) / 24.0 / 3600.0;
      const auto_resp = alloc_values.auto_resp / 24.0 / 3600.0;
      const nee = hetero_resp + auto_resp - gpp_avg;

      cleaf = alloc_values.cleaf;
      croot = alloc_values.croot;
      cstem = alloc_values.cstem;
      cgrain = alloc_values.cgrain;
      litter_cleaf = alloc_values.litter_cleaf;
      litter_croot = alloc_values.litter_croot;
      compost = alloc_values.compost;
      above = alloc_values.above;
      below = alloc_values.below;
      yield_ = alloc_values.yield_;
      grain_fill = alloc_values.grain_fill;
      lai_alloc = alloc_values.lai;
      pheno = alloc_values.pheno;
      lai_prev = lai;

      outputs.gpp_avg.push(gpp_avg);
      outputs.nee.push(nee);
      outputs.hetero_resp.push(hetero_resp);
      outputs.auto_resp.push(auto_resp);
      outputs.cleaf.push(cleaf);
      outputs.croot.push(croot);
      outputs.cstem.push(cstem);
      outputs.cgrain.push(cgrain);
      outputs.lai_alloc.push(lai_alloc);
      outputs.litter_cleaf.push(litter_cleaf);
      outputs.litter_croot.push(litter_croot);
      outputs.soc_total.push(cstate.reduce((sum, value) => sum + value, 0.0));
      outputs.wliq.push(sw_state.Wliq);
      outputs.psi.push(sw_state.Psi);
      outputs.cstate.push([...cstate]);
      outputs.et_total.push(et_acc);
    }

    const final_carry: DailyCarry = {
      cw_state: {
        CanopyStorage: np.array(cw_state.CanopyStorage),
        SWE: np.array(cw_state.SWE),
        swe_i: np.array(cw_state.swe_i),
        swe_l: np.array(cw_state.swe_l),
      },
      sw_state: {
        WatSto: np.array(sw_state.WatSto),
        PondSto: np.array(sw_state.PondSto),
        MaxWatSto: np.array(sw_state.MaxWatSto),
        MaxPondSto: np.array(sw_state.MaxPondSto),
        FcSto: np.array(sw_state.FcSto),
        Wliq: np.array(sw_state.Wliq),
        Psi: np.array(sw_state.Psi),
        Sat: np.array(sw_state.Sat),
        Kh: np.array(sw_state.Kh),
        beta: np.array(sw_state.beta),
      },
      met_rolling: np.array(met_rolling),
      is_first_met: np.array(is_first_met),
      cleaf: np.array(cleaf),
      croot: np.array(croot),
      cstem: np.array(cstem),
      cgrain: np.array(cgrain),
      litter_cleaf: np.array(litter_cleaf),
      litter_croot: np.array(litter_croot),
      compost: np.array(compost),
      soluble: np.array(soluble),
      above: np.array(above),
      below: np.array(below),
      yield_: np.array(yield_),
      grain_fill: np.array(grain_fill),
      lai_alloc: np.array(lai_alloc),
      pheno: np.array(pheno),
      cstate: np.array(cstate),
      nstate: np.array(nstate),
      lai_prev: np.array(lai_prev),
    };

    const daily_output: DailyOutput = {
      gpp_avg: np.array(outputs.gpp_avg),
      nee: np.array(outputs.nee),
      hetero_resp: np.array(outputs.hetero_resp),
      auto_resp: np.array(outputs.auto_resp),
      cleaf: np.array(outputs.cleaf),
      croot: np.array(outputs.croot),
      cstem: np.array(outputs.cstem),
      cgrain: np.array(outputs.cgrain),
      lai_alloc: np.array(outputs.lai_alloc),
      litter_cleaf: np.array(outputs.litter_cleaf),
      litter_croot: np.array(outputs.litter_croot),
      soc_total: np.array(outputs.soc_total),
      wliq: np.array(outputs.wliq),
      psi: np.array(outputs.psi),
      cstate: np.array(outputs.cstate),
      et_total: np.array(outputs.et_total),
    };

    return [final_carry, daily_output];
  } finally {
    tree.dispose(soil_params);
    tree.dispose(aero_params);
    tree.dispose(cs_params);
    tree.dispose(allocation_params);
    yasso_param.dispose();
  }
}

export function runIntegrationScanExperimental(inputs: RunIntegrationInputs): [DailyCarry, DailyOutput] {
  if (
    inputs.phydro_optimizer != null
    && inputs.phydro_optimizer !== "projected_lbfgs"
    && inputs.phydro_optimizer !== "projected_newton"
  ) {
    throw new Error(
      `Unknown phydro optimizer ${inputs.phydro_optimizer}; expected projected_lbfgs or projected_newton`,
    );
  }
  const solverKind = inputs.phydro_optimizer ?? "projected_lbfgs";
  const ndays = leadingLength(inputs.daily_lai);

  const soil_params = tree.makeDisposable({
    watsat: np.array(inputs.watsat),
    watres: np.array(inputs.watres),
    alphaVan: np.array(inputs.alpha_van),
    nVan: np.array(inputs.n_van),
    ksat: np.array(inputs.ksat),
  }) as SoilHydroParams;
  const aero_params = tree.makeDisposable({
    hc: np.array(inputs.hc),
    zmeas: np.array(inputs.zmeas),
    zground: np.array(inputs.zground),
    zo_ground: np.array(inputs.zo_ground),
    w_leaf: np.array(inputs.w_leaf),
  }) as SpafhyAeroParams;
  const cs_params = tree.makeDisposable({
    wmax: np.array(inputs.wmax),
    wmaxsnow: np.array(inputs.wmaxsnow),
    kmelt: np.array(inputs.kmelt),
    kfreeze: np.array(inputs.kfreeze),
    fracSnowliq: np.array(inputs.frac_snowliq),
    gsoil: np.array(inputs.gsoil),
  }) as CanopySnowParams;
  const yasso_param = np.array(inputs.yasso_param);
  const allocation_params = tree.makeDisposable({
    cratio_resp: np.array(inputs.cratio_resp),
    cratio_leaf: np.array(inputs.cratio_leaf),
    cratio_root: np.array(inputs.cratio_root),
    cratio_biomass: np.array(inputs.cratio_biomass),
    harvest_index: np.array(inputs.harvest_index),
    turnover_cleaf: np.array(inputs.turnover_cleaf),
    turnover_croot: np.array(inputs.turnover_croot),
    sla: np.array(inputs.sla),
    q10: np.array(inputs.q10),
    invert_option: np.array(inputs.invert_option),
    pft_is_oat: np.array(inputs.pft_is_oat),
  }) as {
    cratio_resp: np.Array;
    cratio_leaf: np.Array;
    cratio_root: np.Array;
    cratio_biomass: np.Array;
    harvest_index: np.Array;
    turnover_cleaf: np.Array;
    turnover_croot: np.Array;
    sla: np.Array;
    q10: np.Array;
    invert_option: np.Array;
    pft_is_oat: np.Array;
  };

  try {
    const [cw_init, sw_init] = initializationSpafhy(
      inputs.soil_depth,
      inputs.max_poros,
      inputs.fc,
      inputs.maxpond,
      soil_params,
    );

    using yasso_totc = np.array(inputs.yasso_totc);
    using yasso_cn_input = np.array(inputs.yasso_cn_input);
    using yasso_fract_root = np.array(inputs.yasso_fract_root);
    using yasso_fract_legacy = np.array(inputs.yasso_fract_legacy);
    using yasso_tempr_c = np.array(inputs.yasso_tempr_c);
    using yasso_precip_day = np.array(inputs.yasso_precip_day);
    using yasso_tempr_ampl = np.array(inputs.yasso_tempr_ampl);
    const yasso_init = initializeTotcFn(
      yasso_param,
      yasso_totc,
      yasso_cn_input,
      yasso_fract_root,
      yasso_fract_legacy,
      yasso_tempr_c,
      yasso_precip_day,
      yasso_tempr_ampl,
    );

    using max_poros_arr = np.array(inputs.max_poros);
    using rdark_arr = np.array(inputs.rdark);
    using conductivity_arr = np.array(inputs.conductivity);
    using psi50_arr = np.array(inputs.psi50);
    using b_param_arr = np.array(inputs.b_param);
    using alpha_cost_arr = np.array(inputs.alpha_cost);
    using gamma_cost_arr = np.array(inputs.gamma_cost);
    const daily_step = makeDailyStep(
      aero_params,
      cs_params,
      soil_params,
      max_poros_arr,
      rdark_arr,
      conductivity_arr,
      psi50_arr,
      b_param_arr,
      alpha_cost_arr,
      gamma_cost_arr,
      allocation_params.cratio_resp,
      allocation_params.cratio_leaf,
      allocation_params.cratio_root,
      allocation_params.cratio_biomass,
      allocation_params.harvest_index,
      allocation_params.turnover_cleaf,
      allocation_params.turnover_croot,
      allocation_params.sla,
      allocation_params.q10,
      allocation_params.invert_option,
      yasso_param,
      allocation_params.pft_is_oat,
      solverKind,
    );

    const daily_init: DailyCarry = {
      cw_state: cw_init,
      sw_state: sw_init,
      met_rolling: np.array([0.0, 0.0]),
      is_first_met: np.array(true),
      cleaf: np.array(0.0),
      croot: np.array(0.0),
      cstem: np.array(0.0),
      cgrain: np.array(0.0),
      litter_cleaf: np.array(0.0),
      litter_croot: np.array(0.0),
      compost: np.array(0.0),
      soluble: np.array(0.0),
      above: np.array(0.0),
      below: np.array(0.0),
      yield_: np.array(0.0),
      grain_fill: np.array(0.0),
      lai_alloc: np.array(0.0),
      pheno: np.array(1.0),
      cstate: yasso_init.cstate.ref,
      nstate: yasso_init.nstate.ref,
      lai_prev: np.array(0.0),
    };
    yasso_init.cstate.dispose();
    yasso_init.nstate.dispose();

    const daily_forcing: DailyForcing = {
      hourly_temp: np.array(inputs.hourly_temp),
      hourly_rg: np.array(inputs.hourly_rg),
      hourly_prec: np.array(inputs.hourly_prec),
      hourly_vpd: np.array(inputs.hourly_vpd),
      hourly_pres: np.array(inputs.hourly_pres),
      hourly_co2: np.array(inputs.hourly_co2),
      hourly_wind: np.array(inputs.hourly_wind),
      lai: np.array(inputs.daily_lai),
      management_type: np.array(inputs.daily_manage_type),
      management_c_in: np.array(inputs.daily_manage_c_in),
      management_c_out: np.array(inputs.daily_manage_c_out),
    };

    return lax.scan(daily_step, daily_init, daily_forcing, { length: ndays });
  } finally {
    tree.dispose(soil_params);
    tree.dispose(aero_params);
    tree.dispose(cs_params);
    tree.dispose(allocation_params);
    yasso_param.dispose();
  }
}