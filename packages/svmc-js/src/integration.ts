import { lax, tree, type JsTree } from "@hamk-uas/jax-js-nonconsuming";
import { np, retainArray } from "./precision.js";
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

interface DailyBiologyCarry {
  cleaf: np.Array;
  croot: np.Array;
  cstem: np.Array;
  cgrain: np.Array;
  litter_cleaf: np.Array;
  grain_fill: np.Array;
  pheno: np.Array;
  cstate: np.Array;
  nstate: np.Array;
  soluble: np.Array;
}

interface DailyBiologyForcing {
  management_type: np.Array;
  management_c_in: np.Array;
  management_c_out: np.Array;
}

interface DailyBiologyUpdate {
  auto_resp: np.Array;
  cleaf: np.Array;
  croot: np.Array;
  cstem: np.Array;
  cgrain: np.Array;
  litter_cleaf: np.Array;
  litter_croot: np.Array;
  compost: np.Array;
  lai_alloc: np.Array;
  above: np.Array;
  below: np.Array;
  yield_: np.Array;
  grain_fill: np.Array;
  pheno: np.Array;
  hetero_resp: np.Array;
  nee: np.Array;
  soc_total: np.Array;
  new_cstate: np.Array;
  new_nstate: np.Array;
}

interface AllocationParams {
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
}

interface SharedIntegrationResources {
  soil_params: SoilHydroParams;
  aero_params: SpafhyAeroParams;
  cs_params: CanopySnowParams;
  yasso_param: np.Array;
  allocation_params: AllocationParams;
  initial_cw_state: CanopyWaterState;
  initial_sw_state: SoilWaterState;
  initial_cstate: np.Array;
  initial_nstate: np.Array;
}

interface ScalarCanopyWaterState {
  CanopyStorage: number;
  SWE: number;
  swe_i: number;
  swe_l: number;
}

interface ScalarSoilWaterState {
  WatSto: number;
  PondSto: number;
  MaxWatSto: number;
  MaxPondSto: number;
  FcSto: number;
  Wliq: number;
  Psi: number;
  Sat: number;
  Kh: number;
  beta: number;
}

interface ScalarDailyCarryState {
  cw_state: ScalarCanopyWaterState;
  sw_state: ScalarSoilWaterState;
  met_rolling: number[];
  is_first_met: boolean;
  cleaf: number;
  croot: number;
  cstem: number;
  cgrain: number;
  litter_cleaf: number;
  litter_croot: number;
  compost: number;
  soluble: number;
  above: number;
  below: number;
  yield_: number;
  grain_fill: number;
  lai_alloc: number;
  pheno: number;
  cstate: number[];
  nstate: number;
  lai_prev: number;
}

interface EagerDailyOutputBuffers {
  gpp_avg: number[];
  nee: number[];
  hetero_resp: number[];
  auto_resp: number[];
  cleaf: number[];
  croot: number[];
  cstem: number[];
  cgrain: number[];
  lai_alloc: number[];
  litter_cleaf: number[];
  litter_croot: number[];
  soc_total: number[];
  wliq: number[];
  psi: number[];
  cstate: number[][];
  et_total: number[];
}

interface EagerDailyOutputRow {
  gpp_avg: number;
  nee: number;
  hetero_resp: number;
  auto_resp: number;
  cleaf: number;
  croot: number;
  cstem: number;
  cgrain: number;
  lai_alloc: number;
  litter_cleaf: number;
  litter_croot: number;
  soc_total: number;
  wliq: number;
  psi: number;
  cstate: number[];
  et_total: number;
}

interface EagerDailyTransition {
  next_state: ScalarDailyCarryState;
  output_row: EagerDailyOutputRow;
}

interface ScalarDailyBiologyUpdate {
  auto_resp: number;
  cleaf: number;
  croot: number;
  cstem: number;
  cgrain: number;
  litter_cleaf: number;
  litter_croot: number;
  compost: number;
  lai_alloc: number;
  above: number;
  below: number;
  yield_: number;
  grain_fill: number;
  pheno: number;
  hetero_resp: number;
  nee: number;
  soc_total: number;
  new_cstate: number[];
  new_nstate: number;
}

interface ScanDailyTransition {
  next_carry: DailyCarry;
  output: DailyOutput;
}

interface EagerDailyStepInputs {
  hourly_temp: number[][];
  hourly_rg: number[][];
  hourly_prec: number[][];
  hourly_vpd: number[][];
  hourly_pres: number[][];
  hourly_co2: number[][];
  hourly_wind: number[][];
  daily_lai: number[];
  daily_manage_type: number[];
  daily_manage_c_in: number[];
  daily_manage_c_out: number[];
  aero_params: SpafhyAeroParams;
  cs_params: CanopySnowParams;
  soil_params: SoilHydroParams;
  allocation_params: AllocationParams;
  yasso_param: np.Array;
  conductivity_val: number;
  psi50_val: number;
  b_param_val: number;
  alpha_cost_val: number;
  gamma_cost_val: number;
  rdark_val: number;
  max_poros_val: number;
  solverKind: "projected_lbfgs" | "projected_newton";
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

function scalarizeCanopyWaterState(state: CanopyWaterState): ScalarCanopyWaterState {
  return {
    CanopyStorage: scalar(state.CanopyStorage),
    SWE: scalar(state.SWE),
    swe_i: scalar(state.swe_i),
    swe_l: scalar(state.swe_l),
  };
}

function materializeScalarCanopyWaterState(state: ScalarCanopyWaterState): CanopyWaterState {
  return {
    CanopyStorage: np.array(state.CanopyStorage),
    SWE: np.array(state.SWE),
    swe_i: np.array(state.swe_i),
    swe_l: np.array(state.swe_l),
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

function scalarizeSoilWaterState(state: SoilWaterState): ScalarSoilWaterState {
  return {
    WatSto: scalar(state.WatSto),
    PondSto: scalar(state.PondSto),
    MaxWatSto: scalar(state.MaxWatSto),
    MaxPondSto: scalar(state.MaxPondSto),
    FcSto: scalar(state.FcSto),
    Wliq: scalar(state.Wliq),
    Psi: scalar(state.Psi),
    Sat: scalar(state.Sat),
    Kh: scalar(state.Kh),
    beta: scalar(state.beta),
  };
}

function materializeScalarSoilWaterState(state: ScalarSoilWaterState): SoilWaterState {
  return {
    WatSto: np.array(state.WatSto),
    PondSto: np.array(state.PondSto),
    MaxWatSto: np.array(state.MaxWatSto),
    MaxPondSto: np.array(state.MaxPondSto),
    FcSto: np.array(state.FcSto),
    Wliq: np.array(state.Wliq),
    Psi: np.array(state.Psi),
    Sat: np.array(state.Sat),
    Kh: np.array(state.Kh),
    beta: np.array(state.beta),
  };
}

function materializeScalarDailyCarry(state: ScalarDailyCarryState): DailyCarry {
  return {
    cw_state: materializeScalarCanopyWaterState(state.cw_state),
    sw_state: materializeScalarSoilWaterState(state.sw_state),
    met_rolling: np.array(state.met_rolling),
    is_first_met: np.array(state.is_first_met),
    cleaf: np.array(state.cleaf),
    croot: np.array(state.croot),
    cstem: np.array(state.cstem),
    cgrain: np.array(state.cgrain),
    litter_cleaf: np.array(state.litter_cleaf),
    litter_croot: np.array(state.litter_croot),
    compost: np.array(state.compost),
    soluble: np.array(state.soluble),
    above: np.array(state.above),
    below: np.array(state.below),
    yield_: np.array(state.yield_),
    grain_fill: np.array(state.grain_fill),
    lai_alloc: np.array(state.lai_alloc),
    pheno: np.array(state.pheno),
    cstate: np.array(state.cstate),
    nstate: np.array(state.nstate),
    lai_prev: np.array(state.lai_prev),
  };
}

function materializeEagerDailyOutput(outputs: EagerDailyOutputBuffers): DailyOutput {
  return {
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
}

function appendEagerDailyOutput(
  outputs: EagerDailyOutputBuffers,
  row: EagerDailyOutputRow,
): void {
  outputs.gpp_avg.push(row.gpp_avg);
  outputs.nee.push(row.nee);
  outputs.hetero_resp.push(row.hetero_resp);
  outputs.auto_resp.push(row.auto_resp);
  outputs.cleaf.push(row.cleaf);
  outputs.croot.push(row.croot);
  outputs.cstem.push(row.cstem);
  outputs.cgrain.push(row.cgrain);
  outputs.lai_alloc.push(row.lai_alloc);
  outputs.litter_cleaf.push(row.litter_cleaf);
  outputs.litter_croot.push(row.litter_croot);
  outputs.soc_total.push(row.soc_total);
  outputs.wliq.push(row.wliq);
  outputs.psi.push(row.psi);
  outputs.cstate.push([...row.cstate]);
  outputs.et_total.push(row.et_total);
}

function buildScanDailyTransition(
  carry: DailyCarry,
  forcing: DailyForcing,
  final_hourly: HourlyCarry,
  biology_update: DailyBiologyUpdate,
  gpp_avg: np.Array,
): ScanDailyTransition {
  return {
    next_carry: {
      cw_state: {
        CanopyStorage: final_hourly.cw_state.CanopyStorage,
        SWE: final_hourly.cw_state.SWE,
        swe_i: final_hourly.cw_state.swe_i,
        swe_l: final_hourly.cw_state.swe_l,
      },
      sw_state: {
        WatSto: final_hourly.sw_state.WatSto,
        PondSto: final_hourly.sw_state.PondSto,
        MaxWatSto: final_hourly.sw_state.MaxWatSto,
        MaxPondSto: final_hourly.sw_state.MaxPondSto,
        FcSto: final_hourly.sw_state.FcSto,
        Wliq: final_hourly.sw_state.Wliq,
        Psi: final_hourly.sw_state.Psi,
        Sat: final_hourly.sw_state.Sat,
        Kh: final_hourly.sw_state.Kh,
        beta: final_hourly.sw_state.beta,
      },
      met_rolling: final_hourly.met_rolling,
      is_first_met: final_hourly.is_first_met,
      cleaf: biology_update.cleaf,
      croot: biology_update.croot,
      cstem: biology_update.cstem,
      cgrain: biology_update.cgrain,
      litter_cleaf: biology_update.litter_cleaf,
      litter_croot: biology_update.litter_croot,
      compost: biology_update.compost,
      soluble: carry.soluble,
      above: biology_update.above,
      below: biology_update.below,
      yield_: biology_update.yield_,
      grain_fill: biology_update.grain_fill,
      lai_alloc: biology_update.lai_alloc,
      pheno: biology_update.pheno,
      cstate: biology_update.new_cstate,
      nstate: biology_update.new_nstate,
      lai_prev: forcing.lai,
    },
    output: {
      gpp_avg,
      nee: biology_update.nee,
      hetero_resp: biology_update.hetero_resp,
      auto_resp: biology_update.auto_resp,
      cleaf: biology_update.cleaf,
      croot: biology_update.croot,
      cstem: biology_update.cstem,
      cgrain: biology_update.cgrain,
      lai_alloc: biology_update.lai_alloc,
      litter_cleaf: biology_update.litter_cleaf,
      litter_croot: biology_update.litter_croot,
      soc_total: biology_update.soc_total,
      wliq: final_hourly.sw_state.Wliq,
      psi: final_hourly.sw_state.Psi,
      cstate: biology_update.new_cstate,
      et_total: final_hourly.et_acc,
    },
  };
}

function scalarizeDailyBiologyUpdate(update: DailyBiologyUpdate): ScalarDailyBiologyUpdate {
  return {
    auto_resp: scalar(update.auto_resp),
    cleaf: scalar(update.cleaf),
    croot: scalar(update.croot),
    cstem: scalar(update.cstem),
    cgrain: scalar(update.cgrain),
    litter_cleaf: scalar(update.litter_cleaf),
    litter_croot: scalar(update.litter_croot),
    compost: scalar(update.compost),
    lai_alloc: scalar(update.lai_alloc),
    above: scalar(update.above),
    below: scalar(update.below),
    yield_: scalar(update.yield_),
    grain_fill: scalar(update.grain_fill),
    pheno: scalar(update.pheno),
    hetero_resp: scalar(update.hetero_resp),
    nee: scalar(update.nee),
    soc_total: scalar(update.soc_total),
    new_cstate: update.new_cstate.js() as number[],
    new_nstate: scalar(update.new_nstate),
  };
}

function buildEagerDailyTransition(
  state: ScalarDailyCarryState,
  cw_state: ScalarCanopyWaterState,
  sw_state: ScalarSoilWaterState,
  met_rolling: number[],
  is_first_met: boolean,
  lai: number,
  gpp_avg: number,
  et_total: number,
  biology: ScalarDailyBiologyUpdate,
): EagerDailyTransition {
  return {
    next_state: {
      cw_state,
      sw_state,
      met_rolling,
      is_first_met,
      cleaf: biology.cleaf,
      croot: biology.croot,
      cstem: biology.cstem,
      cgrain: biology.cgrain,
      litter_cleaf: biology.litter_cleaf,
      litter_croot: biology.litter_croot,
      compost: biology.compost,
      soluble: state.soluble,
      above: biology.above,
      below: biology.below,
      yield_: biology.yield_,
      grain_fill: biology.grain_fill,
      lai_alloc: biology.lai_alloc,
      pheno: biology.pheno,
      cstate: biology.new_cstate,
      nstate: biology.new_nstate,
      lai_prev: lai,
    },
    output_row: {
      gpp_avg,
      nee: biology.nee,
      hetero_resp: biology.hetero_resp,
      auto_resp: biology.auto_resp,
      cleaf: biology.cleaf,
      croot: biology.croot,
      cstem: biology.cstem,
      cgrain: biology.cgrain,
      lai_alloc: biology.lai_alloc,
      litter_cleaf: biology.litter_cleaf,
      litter_croot: biology.litter_croot,
      soc_total: biology.soc_total,
      wliq: sw_state.Wliq,
      psi: sw_state.Psi,
      cstate: biology.new_cstate,
      et_total,
    },
  };
}

function runEagerDailyStep(
  state: ScalarDailyCarryState,
  day_idx: number,
  inputs: EagerDailyStepInputs,
): EagerDailyTransition {
  const lai = inputs.daily_lai[day_idx];
  const fapar = 1.0 - Math.exp(-K_EXT * lai);
  let temp_acc = 0.0;
  let precip_acc = 0.0;
  let gpp_acc = 0.0;
  let vcmax_acc = 0.0;
  let num_gpp = 0.0;
  let num_vcmax = 0.0;
  let et_acc = 0.0;

  let cw_state = state.cw_state;
  let sw_state = state.sw_state;
  let met_rolling = state.met_rolling;
  let is_first_met = state.is_first_met;

  for (let hour_idx = 0; hour_idx < 24; hour_idx += 1) {
    const temp_k = inputs.hourly_temp[day_idx][hour_idx];
    const rg = inputs.hourly_rg[day_idx][hour_idx];
    const prec = inputs.hourly_prec[day_idx][hour_idx];
    const vpd = inputs.hourly_vpd[day_idx][hour_idx];
    const pres = inputs.hourly_pres[day_idx][hour_idx];
    const co2 = inputs.hourly_co2[day_idx][hour_idx];
    const wind = inputs.hourly_wind[day_idx][hour_idx];
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
        inputs.rdark_val,
        inputs.conductivity_val,
        inputs.psi50_val,
        inputs.b_param_val,
        inputs.alpha_cost_val,
        inputs.gamma_cost_val,
        KPHIO,
        inputs.solverKind,
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

    const gpp_hr = (aj + inputs.rdark_val * vcmax_hr) * C_MOLMASS * 1.0e-6 * 1.0e-3 * lai;

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
      const cw_input = materializeScalarCanopyWaterState(cw_state);
      const sw_input = materializeScalarSoilWaterState(sw_state);
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
        inputs.aero_params,
        inputs.cs_params,
        time_step,
      );
      using potinf = cw_flux.PotInfiltration.ref;
      using soil_evap = cw_flux.SoilEvap.ref;
      using latflow = np.array(0.0);
      using tr_arr = np.array(tr_spafhy);
      using max_poros_arr = np.array(inputs.max_poros_val);
      const soil_result = soilWater(
        sw_input,
        inputs.soil_params,
        max_poros_arr,
        potinf,
        tr_arr,
        soil_evap,
        latflow,
        time_step,
      );

      cw_state = scalarizeCanopyWaterState(cw_output);
      sw_state = scalarizeSoilWaterState(soil_result.state);

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
  const leaf_rdark_day = inputs.rdark_val * vcmax_avg * C_MOLMASS * 1.0e-6 * 1.0e-3 * lai;
  const delta_lai = lai - state.lai_prev;

  let daily_update: DailyBiologyUpdate | null = null;
  {
    using delta_lai_arr = np.array(delta_lai);
    using temp_avg_arr = np.array(temp_avg);
    using gpp_avg_arr = np.array(gpp_avg);
    using leaf_rdark_day_arr = np.array(leaf_rdark_day);
    using precip_acc_arr = np.array(precip_acc);
    using cleaf_arr = np.array(state.cleaf);
    using croot_arr = np.array(state.croot);
    using cstem_arr = np.array(state.cstem);
    using cgrain_arr = np.array(state.cgrain);
    using litter_cleaf_arr = np.array(state.litter_cleaf);
    using grain_fill_arr = np.array(state.grain_fill);
    using pheno_arr = np.array(state.pheno);
    using cstate_arr = np.array(state.cstate);
    using nstate_arr = np.array(state.nstate);
    using soluble_arr = np.array(state.soluble);
    using manage_type_arr = np.array(inputs.daily_manage_type[day_idx]);
    using manage_c_in_arr = np.array(inputs.daily_manage_c_in[day_idx]);
    using manage_c_out_arr = np.array(inputs.daily_manage_c_out[day_idx]);
    daily_update = runDailyBiologyStep(
      {
        cleaf: cleaf_arr,
        croot: croot_arr,
        cstem: cstem_arr,
        cgrain: cgrain_arr,
        litter_cleaf: litter_cleaf_arr,
        grain_fill: grain_fill_arr,
        pheno: pheno_arr,
        cstate: cstate_arr,
        nstate: nstate_arr,
        soluble: soluble_arr,
      },
      {
        management_type: manage_type_arr,
        management_c_in: manage_c_in_arr,
        management_c_out: manage_c_out_arr,
      },
      delta_lai_arr,
      temp_avg_arr,
      gpp_avg_arr,
      leaf_rdark_day_arr,
      precip_acc_arr,
      inputs.allocation_params.cratio_resp,
      inputs.allocation_params.cratio_leaf,
      inputs.allocation_params.cratio_root,
      inputs.allocation_params.cratio_biomass,
      inputs.allocation_params.harvest_index,
      inputs.allocation_params.turnover_cleaf,
      inputs.allocation_params.turnover_croot,
      inputs.allocation_params.sla,
      inputs.allocation_params.q10,
      inputs.allocation_params.invert_option,
      inputs.yasso_param,
      inputs.allocation_params.pft_is_oat,
    );

    const scalar_biology = scalarizeDailyBiologyUpdate(daily_update);
    const transition = buildEagerDailyTransition(
      state,
      cw_state,
      sw_state,
      met_rolling,
      is_first_met,
      lai,
      gpp_avg,
      et_acc,
      scalar_biology,
    );

    tree.dispose(daily_update);
    return transition;
  }
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

function createSharedIntegrationResources(
  inputs: RunIntegrationInputs,
): SharedIntegrationResources {
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
  }) as AllocationParams;

  const [initial_cw_state, initial_sw_state] = initializationSpafhy(
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

  return {
    soil_params,
    aero_params,
    cs_params,
    yasso_param,
    allocation_params,
    initial_cw_state,
    initial_sw_state,
    initial_cstate: yasso_init.cstate,
    initial_nstate: yasso_init.nstate,
  };
}

function disposeSharedIntegrationResources(resources: SharedIntegrationResources): void {
  tree.dispose(resources.soil_params);
  tree.dispose(resources.aero_params);
  tree.dispose(resources.cs_params);
  tree.dispose(resources.allocation_params);
  resources.yasso_param.dispose();
}

function runDailyBiologyStep(
  carry: DailyBiologyCarry,
  forcing: DailyBiologyForcing,
  delta_lai: np.Array,
  temp_avg: np.Array,
  gpp_avg: np.Array,
  leaf_rdark_day: np.Array,
  precip_acc: np.Array,
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
): DailyBiologyUpdate {
  using inv_result = tree.makeDisposable(invertAllocFn(
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
  )) as {
    deltaLai: np.Array;
    litterCleaf: np.Array;
    cleaf: np.Array;
    cratioLeaf: np.Array;
    cratioRoot: np.Array;
    turnoverCleaf: np.Array;
  };

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
  using alloc_npp_day = alloc_result.nppDay;
  using alloc_auto_resp = alloc_result.autoResp;

  using input_cfract = inputsToFractions(
    alloc_result.litterCleaf,
    alloc_result.litterCroot,
    carry.soluble,
    alloc_result.compost,
  );
  using one_day = np.array(1.0);
  using decomp_result = tree.makeDisposable(decomposeFn(
    yasso_param,
    one_day,
    temp_avg,
    precip_acc,
    carry.cstate,
    carry.nstate,
  )) as { ctend: np.Array; ntend: np.Array };
  using cstate_plus_ctend = carry.cstate.add(decomp_result.ctend);
  const new_cstate = cstate_plus_ctend.add(input_cfract);
  const new_nstate = carry.nstate.add(decomp_result.ntend);

  using ctend_neg = decomp_result.ctend.neg();
  using hetero_resp_num = np.sum(ctend_neg);
  using hetero_resp_day = hetero_resp_num.div(24.0);
  const hetero_resp = hetero_resp_day.div(3600.0);
  using auto_resp_day = alloc_auto_resp.div(24.0);
  const auto_resp = auto_resp_day.div(3600.0);
  using total_resp = hetero_resp.add(auto_resp);
  const nee = total_resp.sub(gpp_avg);
  const soc_total = np.sum(new_cstate);

  return {
    auto_resp,
    cleaf: alloc_result.cleaf,
    croot: alloc_result.croot,
    cstem: alloc_result.cstem,
    cgrain: alloc_result.cgrain,
    litter_cleaf: alloc_result.litterCleaf,
    litter_croot: alloc_result.litterCroot,
    compost: alloc_result.compost,
    lai_alloc: alloc_result.lai,
    above: alloc_result.abovebiomass,
    below: alloc_result.belowbiomass,
    yield_: alloc_result.yield,
    grain_fill: alloc_result.grainFill,
    pheno: alloc_result.phenoStage,
    hetero_resp,
    nee,
    soc_total,
    new_cstate,
    new_nstate,
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
  using soilDepth = retainArray(soil_depth);
  using maxPoros = retainArray(max_poros);
  using fcArr = retainArray(fc);
  using maxpondArr = retainArray(maxpond);

  using maxWatStoBase = soilDepth.mul(maxPoros);
  using maxWatSto = maxWatStoBase.mul(1000.0);
  using fcStoBase = soilDepth.mul(fcArr);
  using fcSto = fcStoBase.mul(1000.0);
  using watSto = maxWatSto.mul(0.9);
  const pondSto = np.array(0.0);
  using watStoRatio = watSto.div(maxWatSto);
  using satScale = np.minimum(1.0, watStoRatio);
  using wliq = maxPoros.mul(satScale);
  using sat = wliq.div(maxPoros);
  using beta = np.minimum(1.0, sat);
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
  const time_step = lai.mul(0.0).add(TIME_STEP);
  const zero_latflow = lai.mul(0.0);

  return (carry: HourlyCarry, forcing: HourlyForcing): [HourlyCarry, null] => {
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
    const { aj, gs, vcmax: vcmax_hr } = phydro;

    using gpp_inner = rdark.mul(vcmax_hr);
    using gpp_sum = aj.add(gpp_inner);
    using gpp_scaled0 = gpp_sum.mul(C_MOLMASS);
    using gpp_scaled1 = gpp_scaled0.mul(1.0e-6);
    using gpp_scaled2 = gpp_scaled1.mul(1.0e-3);
    using gpp_scaled = gpp_scaled2.mul(lai);
    using lai_positive = lai.greater(LAI_GUARD);
    using rg_positive = forcing.rg.greater(0.0);
    using has_light = lai_positive.mul(rg_positive);
    const gpp_hr = np.where(has_light, gpp_scaled, 0.0);
    const aj_guarded = np.where(has_light, aj, 0.0);
    const gs_guarded = np.where(has_light, gs, 0.0);
    const vcmax_guarded = np.where(has_light, vcmax_hr, 0.0);
    tree.dispose(phydro);

    using rho_w = densityH2o(tc, forcing.pres);
    using vpd_over_pres = forcing.vpd.div(forcing.pres);
    using tr_scaled0 = gs_guarded.mul(1.6);
    using tr_scaled1 = tr_scaled0.mul(vpd_over_pres);
    using tr_scaled = tr_scaled1.mul(H2O_MOLMASS);
    using tr_raw_base = tr_scaled.div(rho_w);
    using tr_raw = tr_raw_base.mul(lai);
    using gs_is_finite = np.isfinite(gs_guarded);
    using tr_valid = gs_is_finite.mul(has_light);
    const tr_phydro = np.where(tr_valid, tr_raw, 0.0);

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
    const et_hr = et_from_canopy.add(tr_spafhy);

    const soil_result = soilWater(
      carry.sw_state,
      soil_params,
      max_poros,
      cw_flux.PotInfiltration,
      tr_spafhy,
      cw_flux.SoilEvap,
      zero_latflow,
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
    tree.dispose(cw_flux);

    using gpp_valid_finite = np.isfinite(aj_guarded);
    using gpp_valid_nonneg = aj_guarded.greaterEqual(0.0);
    using gpp_valid = gpp_valid_finite.mul(gpp_valid_nonneg);
    const gpp_to_add = np.where(gpp_valid, gpp_hr, 0.0);
    using gpp_count_inc = np.where(gpp_valid, 1.0, 0.0);
    const num_gpp = carry.num_gpp.add(gpp_count_inc);

    using vcmax_valid_finite = np.isfinite(vcmax_guarded);
    using vcmax_valid_positive = vcmax_guarded.greater(0.0);
    using vcmax_valid = vcmax_valid_finite.mul(vcmax_valid_positive);
    const vcmax_to_add = np.where(vcmax_valid, vcmax_guarded, 0.0);
    using vcmax_count_inc = np.where(vcmax_valid, 1.0, 0.0);
    const num_vcmax = carry.num_vcmax.add(vcmax_count_inc);

    using new_met_temp = new_met_rolling.slice(0);
    using new_met_prec = new_met_rolling.slice(1);
    const new_temp_acc = carry.temp_acc.add(new_met_temp);
    using precip_increment = new_met_prec.mul(TIME_STEP * 3600.0);
    const new_precip_acc = carry.precip_acc.add(precip_increment);
    const new_gpp_acc = carry.gpp_acc.add(gpp_to_add);
    const new_vcmax_acc = carry.vcmax_acc.add(vcmax_to_add);
    const new_et_acc = carry.et_acc.add(et_hr);
    const not_first_met = np.where(carry.is_first_met, false, carry.is_first_met);
    const next_sw_state: SoilWaterState = {
      WatSto: soil_result.state.WatSto,
      PondSto: soil_result.state.PondSto,
      MaxWatSto: soil_result.state.MaxWatSto,
      MaxPondSto: soil_result.state.MaxPondSto,
      FcSto: soil_result.state.FcSto,
      Wliq: soil_result.state.Wliq,
      Psi: soil_result.state.Psi,
      Sat: soil_result.state.Sat,
      Kh: soil_result.state.Kh,
      beta: soil_result.state.beta,
    };
    const next_carry: HourlyCarry = {
      cw_state,
      sw_state: next_sw_state,
      met_rolling: new_met_rolling,
      is_first_met: not_first_met,
      temp_acc: new_temp_acc,
      precip_acc: new_precip_acc,
      gpp_acc: new_gpp_acc,
      vcmax_acc: new_vcmax_acc,
      num_gpp,
      num_vcmax,
      et_acc: new_et_acc,
    };

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
    const fapar = one.sub(fapar_exp);

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

    // Seed the nested hourly loop from existing traced scalars instead of
    // constructing fresh anonymous arrays inside the outer scan body.
    using zero_temp_acc = carry.cleaf.mul(0.0);
    using zero_precip_acc = carry.croot.mul(0.0);
    using zero_gpp_acc = carry.cstem.mul(0.0);
    using zero_vcmax_acc = carry.cgrain.mul(0.0);
    using zero_num_gpp = carry.litter_cleaf.mul(0.0);
    using zero_num_vcmax = carry.litter_croot.mul(0.0);
    using zero_et_acc = carry.compost.mul(0.0);

    const hourly_init: HourlyCarry = {
      cw_state: carry.cw_state,
      sw_state: carry.sw_state,
      met_rolling: carry.met_rolling,
      is_first_met: carry.is_first_met,
      temp_acc: zero_temp_acc,
      precip_acc: zero_precip_acc,
      gpp_acc: zero_gpp_acc,
      vcmax_acc: zero_vcmax_acc,
      num_gpp: zero_num_gpp,
      num_vcmax: zero_num_vcmax,
      et_acc: zero_et_acc,
    };

    const hourly_body = (hourIdx: np.Array, hourlyCarry: HourlyCarry): HourlyCarry => {
      const temp_k_slice = lax.dynamicSlice(hourly_forcing.temp_k, [hourIdx], [1]);
      const rg_slice = lax.dynamicSlice(hourly_forcing.rg, [hourIdx], [1]);
      const prec_slice = lax.dynamicSlice(hourly_forcing.prec, [hourIdx], [1]);
      const vpd_slice = lax.dynamicSlice(hourly_forcing.vpd, [hourIdx], [1]);
      const pres_slice = lax.dynamicSlice(hourly_forcing.pres, [hourIdx], [1]);
      const co2_slice = lax.dynamicSlice(hourly_forcing.co2, [hourIdx], [1]);
      const wind_slice = lax.dynamicSlice(hourly_forcing.wind, [hourIdx], [1]);
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

    const final_hourly = lax.foriLoop(
      0,
      24,
      hourly_body as unknown as (i: np.Array, carry: JsTree<np.Array>) => JsTree<np.Array>,
      hourly_init as unknown as JsTree<np.Array>,
    ) as unknown as HourlyCarry;

    using temp_avg = final_hourly.temp_acc.div(24.0);
    using _gpp_ratio = final_hourly.gpp_acc.div(final_hourly.num_gpp);
    const gpp_avg = np.where(
      final_hourly.num_gpp.greater(0.0),
      _gpp_ratio,
      0.0,
    );
    using _vcmax_ratio = final_hourly.vcmax_acc.div(final_hourly.num_vcmax);
    const vcmax_avg = np.where(
      final_hourly.num_vcmax.greater(0.0),
      _vcmax_ratio,
      0.0,
    );
    const precip_acc = final_hourly.precip_acc;
    using leaf_rdark_base = rdark.mul(vcmax_avg);
    using leaf_rdark_scale0 = leaf_rdark_base.mul(C_MOLMASS);
    using leaf_rdark_scale1 = leaf_rdark_scale0.mul(1.0e-6);
    using leaf_rdark_scale = leaf_rdark_scale1.mul(1.0e-3);
    const leaf_rdark_day = leaf_rdark_scale.mul(forcing.lai);

    const biology_update = runDailyBiologyStep(
      carry,
      forcing,
      delta_lai,
      temp_avg,
      gpp_avg,
      leaf_rdark_day,
      precip_acc,
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
      yasso_param,
      pft_is_oat,
    );

    const transition = buildScanDailyTransition(
      carry,
      forcing,
      final_hourly,
      biology_update,
      gpp_avg,
    );

    return [transition.next_carry, transition.output];
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
  const resources = createSharedIntegrationResources(inputs);
  const {
    soil_params,
    aero_params,
    cs_params,
    yasso_param,
    allocation_params,
    initial_cw_state,
    initial_sw_state,
    initial_cstate,
    initial_nstate,
  } = resources;

  try {
    let cw_state = scalarizeCanopyWaterState(initial_cw_state);
    let sw_state = scalarizeSoilWaterState(initial_sw_state);
    disposeCanopyWaterState(initial_cw_state);
    disposeSoilWaterState(initial_sw_state);

    let cstate = initial_cstate.js() as number[];
    let nstate = scalar(initial_nstate);
    initial_cstate.dispose();
    initial_nstate.dispose();

    let state: ScalarDailyCarryState = {
      cw_state,
      sw_state,
      met_rolling: [0.0, 0.0],
      is_first_met: true,
      cleaf: 0.0,
      croot: 0.0,
      cstem: 0.0,
      cgrain: 0.0,
      litter_cleaf: 0.0,
      litter_croot: 0.0,
      compost: 0.0,
      soluble: 0.0,
      above: 0.0,
      below: 0.0,
      yield_: 0.0,
      grain_fill: 0.0,
      lai_alloc: 0.0,
      pheno: 1.0,
      cstate,
      nstate,
      lai_prev: 0.0,
    };

    const outputs: EagerDailyOutputBuffers = {
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

    const eagerDailyInputs: EagerDailyStepInputs = {
      hourly_temp,
      hourly_rg,
      hourly_prec,
      hourly_vpd,
      hourly_pres,
      hourly_co2,
      hourly_wind,
      daily_lai,
      daily_manage_type,
      daily_manage_c_in,
      daily_manage_c_out,
      aero_params,
      cs_params,
      soil_params,
      allocation_params,
      yasso_param,
      conductivity_val,
      psi50_val,
      b_param_val,
      alpha_cost_val,
      gamma_cost_val,
      rdark_val,
      max_poros_val,
      solverKind,
    };

    for (let day_idx = 0; day_idx < ndays; day_idx += 1) {
      const transition = runEagerDailyStep(state, day_idx, eagerDailyInputs);
      state = transition.next_state;
      appendEagerDailyOutput(outputs, transition.output_row);
    }

    const final_carry = materializeScalarDailyCarry(state);

    const daily_output = materializeEagerDailyOutput(outputs);

    return [final_carry, daily_output];
  } finally {
    disposeSharedIntegrationResources(resources);
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
  const resources = createSharedIntegrationResources(inputs);
  const {
    soil_params,
    aero_params,
    cs_params,
    yasso_param,
    allocation_params,
    initial_cw_state,
    initial_sw_state,
    initial_cstate,
    initial_nstate,
  } = resources;

  const disposeIfNotAliased = (initial: np.Array, escaped: np.Array): void => {
    if (initial !== escaped) initial.dispose();
  };

  try {
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
      cw_state: initial_cw_state,
      sw_state: initial_sw_state,
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
      cstate: initial_cstate.ref,
      nstate: initial_nstate.ref,
      lai_prev: np.array(0.0),
    };
    initial_cstate.dispose();
    initial_nstate.dispose();

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

    const result = lax.scan(
      daily_step as unknown as (
        carry: JsTree<np.Array>,
        forcing: JsTree<np.Array>,
      ) => [JsTree<np.Array>, JsTree<np.Array>],
      daily_init as unknown as JsTree<np.Array>,
      daily_forcing as unknown as JsTree<np.Array>,
      { length: ndays },
    ) as unknown as [DailyCarry, DailyOutput];

    const [final_carry] = result;

    // The scan owns its returned carry and outputs, but the large forcing arrays
    // and most initial carry leaves are internal-only and must be released here.
    daily_forcing.hourly_temp.dispose();
    daily_forcing.hourly_rg.dispose();
    daily_forcing.hourly_prec.dispose();
    daily_forcing.hourly_vpd.dispose();
    daily_forcing.hourly_pres.dispose();
    daily_forcing.hourly_co2.dispose();
    daily_forcing.hourly_wind.dispose();
    daily_forcing.lai.dispose();
    daily_forcing.management_type.dispose();
    daily_forcing.management_c_in.dispose();
    daily_forcing.management_c_out.dispose();

    disposeIfNotAliased(
      daily_init.cw_state.CanopyStorage,
      final_carry.cw_state.CanopyStorage,
    );
    disposeIfNotAliased(daily_init.cw_state.SWE, final_carry.cw_state.SWE);
    disposeIfNotAliased(daily_init.cw_state.swe_i, final_carry.cw_state.swe_i);
    disposeIfNotAliased(daily_init.cw_state.swe_l, final_carry.cw_state.swe_l);
    disposeIfNotAliased(daily_init.sw_state.WatSto, final_carry.sw_state.WatSto);
    disposeIfNotAliased(daily_init.sw_state.PondSto, final_carry.sw_state.PondSto);
    disposeIfNotAliased(daily_init.sw_state.MaxWatSto, final_carry.sw_state.MaxWatSto);
    disposeIfNotAliased(
      daily_init.sw_state.MaxPondSto,
      final_carry.sw_state.MaxPondSto,
    );
    disposeIfNotAliased(daily_init.sw_state.FcSto, final_carry.sw_state.FcSto);
    disposeIfNotAliased(daily_init.sw_state.Wliq, final_carry.sw_state.Wliq);
    disposeIfNotAliased(daily_init.sw_state.Psi, final_carry.sw_state.Psi);
    disposeIfNotAliased(daily_init.sw_state.Sat, final_carry.sw_state.Sat);
    disposeIfNotAliased(daily_init.sw_state.Kh, final_carry.sw_state.Kh);
    disposeIfNotAliased(daily_init.sw_state.beta, final_carry.sw_state.beta);
    disposeIfNotAliased(daily_init.met_rolling, final_carry.met_rolling);
    disposeIfNotAliased(daily_init.is_first_met, final_carry.is_first_met);
    disposeIfNotAliased(daily_init.cleaf, final_carry.cleaf);
    disposeIfNotAliased(daily_init.croot, final_carry.croot);
    disposeIfNotAliased(daily_init.cstem, final_carry.cstem);
    disposeIfNotAliased(daily_init.cgrain, final_carry.cgrain);
    disposeIfNotAliased(daily_init.litter_cleaf, final_carry.litter_cleaf);
    disposeIfNotAliased(daily_init.litter_croot, final_carry.litter_croot);
    disposeIfNotAliased(daily_init.compost, final_carry.compost);
    disposeIfNotAliased(daily_init.soluble, final_carry.soluble);
    disposeIfNotAliased(daily_init.above, final_carry.above);
    disposeIfNotAliased(daily_init.below, final_carry.below);
    disposeIfNotAliased(daily_init.yield_, final_carry.yield_);
    disposeIfNotAliased(daily_init.grain_fill, final_carry.grain_fill);
    disposeIfNotAliased(daily_init.lai_alloc, final_carry.lai_alloc);
    disposeIfNotAliased(daily_init.pheno, final_carry.pheno);
    disposeIfNotAliased(daily_init.cstate, final_carry.cstate);
    disposeIfNotAliased(daily_init.nstate, final_carry.nstate);
    disposeIfNotAliased(daily_init.lai_prev, final_carry.lai_prev);

    return result;
  } finally {
    disposeSharedIntegrationResources(resources);
  }
}