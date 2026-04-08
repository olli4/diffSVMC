import { jit, lax, tree, type JsTree } from "@hamk-uas/jax-js-nonconsuming";
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

export type DailyCarry = DailyBoundaryCarry<
  CanopyWaterState,
  SoilWaterState,
  np.Array,
  np.Array,
  np.Array,
  np.Array,
  np.Array
>;

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

type DailyBiologyUpdate = DailyBoundaryBiologyState<np.Array, np.Array>;

type SolverKind = "projected_lbfgs" | "projected_newton";

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

interface IntegrationRunContext {
  solverKind: SolverKind;
  ndays: number;
  resources: SharedIntegrationResources;
}

type IntegrationResult = [DailyCarry, DailyOutput];

type ScanDailySequenceResult = [DailyCarry, DailyOutput];

type ScanDailySequenceFn = (
  daily_init: DailyCarry,
  daily_forcing: DailyForcing,
) => ScanDailySequenceResult;

type ScanDailySequenceRunnerFactory = (
  ndays: number,
  daily_step: (carry: DailyCarry, forcing: DailyForcing) => [DailyCarry, DailyOutput],
) => ScanDailySequenceFn;

interface DailyBoundaryOutput<TValue, TVector> {
  gpp_avg: TValue;
  nee: TValue;
  hetero_resp: TValue;
  auto_resp: TValue;
  cleaf: TValue;
  croot: TValue;
  cstem: TValue;
  cgrain: TValue;
  lai_alloc: TValue;
  litter_cleaf: TValue;
  litter_croot: TValue;
  soc_total: TValue;
  wliq: TValue;
  psi: TValue;
  cstate: TVector;
  et_total: TValue;
}

interface DailyBoundaryTransition<TCarry, TOutput> {
  next_carry: TCarry;
  output: TOutput;
}

interface DailyBoundarySoilState<TValue> {
  Wliq: TValue;
  Psi: TValue;
}

interface DailyBoundaryBiologyState<TValue, TVector> {
  auto_resp: TValue;
  cleaf: TValue;
  croot: TValue;
  cstem: TValue;
  cgrain: TValue;
  litter_cleaf: TValue;
  litter_croot: TValue;
  compost: TValue;
  lai_alloc: TValue;
  above: TValue;
  below: TValue;
  yield_: TValue;
  grain_fill: TValue;
  pheno: TValue;
  hetero_resp: TValue;
  nee: TValue;
  soc_total: TValue;
  cstate: TVector;
  nstate: TValue;
}

interface DailyBoundarySnapshot<
  TCanopyState,
  TSoilState extends DailyBoundarySoilState<TValue>,
  TMetRolling,
  TFlag,
  TLaiPrev,
  TValue,
  TVector,
> {
  cw_state: TCanopyState;
  sw_state: TSoilState;
  met_rolling: TMetRolling;
  is_first_met: TFlag;
  soluble: TValue;
  lai_prev: TLaiPrev;
  gpp_avg: TValue;
  et_total: TValue;
  biology: DailyBoundaryBiologyState<TValue, TVector>;
}

interface DailyBoundaryCarry<
  TCanopyState,
  TSoilState,
  TMetRolling,
  TFlag,
  TLaiPrev,
  TValue,
  TVector,
> {
  cw_state: TCanopyState;
  sw_state: TSoilState;
  met_rolling: TMetRolling;
  is_first_met: TFlag;
  cleaf: TValue;
  croot: TValue;
  cstem: TValue;
  cgrain: TValue;
  litter_cleaf: TValue;
  litter_croot: TValue;
  compost: TValue;
  soluble: TValue;
  above: TValue;
  below: TValue;
  yield_: TValue;
  grain_fill: TValue;
  lai_alloc: TValue;
  pheno: TValue;
  cstate: TVector;
  nstate: TValue;
  lai_prev: TLaiPrev;
}

type ScanDailyTransition = DailyBoundaryTransition<DailyCarry, DailyOutput>;
interface ScanDailyStepResources {
  daily_step: (carry: DailyCarry, forcing: DailyForcing) => [DailyCarry, DailyOutput];
  max_poros: np.Array;
  rdark: np.Array;
  conductivity: np.Array;
  psi50: np.Array;
  b_param: np.Array;
  alpha_cost: np.Array;
  gamma_cost: np.Array;
}

function leadingLength(value: np.ArrayLike): number {
  if (Array.isArray(value)) return value.length;
  const maybeShape = value as { shape?: number[] };
  if (Array.isArray(maybeShape.shape) && maybeShape.shape.length > 0) {
    return maybeShape.shape[0];
  }
  throw new TypeError("Unable to infer leading dimension for integration inputs");
}

function resolveSolverKind(phydro_optimizer?: string): SolverKind {
  if (
    phydro_optimizer != null
    && phydro_optimizer !== "projected_lbfgs"
    && phydro_optimizer !== "projected_newton"
  ) {
    throw new Error(
      `Unknown phydro optimizer ${phydro_optimizer}; expected projected_lbfgs or projected_newton`,
    );
  }
  return phydro_optimizer ?? "projected_lbfgs";
}

function createScanDailyStepResources(
  inputs: RunIntegrationInputs,
  resources: Pick<
    SharedIntegrationResources,
    "soil_params" | "aero_params" | "cs_params" | "allocation_params" | "yasso_param"
  >,
  solverKind: SolverKind,
): ScanDailyStepResources {
  const max_poros = np.array(inputs.max_poros);
  const rdark = np.array(inputs.rdark);
  const conductivity = np.array(inputs.conductivity);
  const psi50 = np.array(inputs.psi50);
  const b_param = np.array(inputs.b_param);
  const alpha_cost = np.array(inputs.alpha_cost);
  const gamma_cost = np.array(inputs.gamma_cost);

  return {
    max_poros,
    rdark,
    conductivity,
    psi50,
    b_param,
    alpha_cost,
    gamma_cost,
    daily_step: makeDailyStep(
      resources.aero_params,
      resources.cs_params,
      resources.soil_params,
      max_poros,
      rdark,
      conductivity,
      psi50,
      b_param,
      alpha_cost,
      gamma_cost,
      resources.allocation_params.cratio_resp,
      resources.allocation_params.cratio_leaf,
      resources.allocation_params.cratio_root,
      resources.allocation_params.cratio_biomass,
      resources.allocation_params.harvest_index,
      resources.allocation_params.turnover_cleaf,
      resources.allocation_params.turnover_croot,
      resources.allocation_params.sla,
      resources.allocation_params.q10,
      resources.allocation_params.invert_option,
      resources.yasso_param,
      resources.allocation_params.pft_is_oat,
      solverKind,
    ),
  };
}

function disposeScanDailyStepResources(resources: ScanDailyStepResources): void {
  resources.max_poros.dispose();
  resources.rdark.dispose();
  resources.conductivity.dispose();
  resources.psi50.dispose();
  resources.b_param.dispose();
  resources.alpha_cost.dispose();
  resources.gamma_cost.dispose();
}

function consumeInitialScanDailyCarry(
  initial_cw_state: CanopyWaterState,
  initial_sw_state: SoilWaterState,
  initial_cstate: np.Array,
  initial_nstate: np.Array,
): DailyCarry {
  const daily_init = createInitialDailyBoundaryCarry(
    initial_cw_state,
    initial_sw_state,
    np.array([0.0, 0.0]),
    np.array(true),
    // jax-js-lint: allow-ref
    initial_cstate.ref,
    // jax-js-lint: allow-ref
    initial_nstate.ref,
    np.array(0.0),
    () => np.array(0.0),
    () => np.array(1.0),
  );
  initial_cstate.dispose();
  initial_nstate.dispose();
  return daily_init;
}

function consumeInitialScanDailyCarryFromResources(resources: SharedIntegrationResources): DailyCarry {
  return consumeInitialScanDailyCarry(
    resources.initial_cw_state,
    resources.initial_sw_state,
    resources.initial_cstate,
    resources.initial_nstate,
  );
}

function executeScanDailySequence(
  daily_init: DailyCarry,
  daily_forcing: DailyForcing,
  ndays: number,
  daily_step: (carry: DailyCarry, forcing: DailyForcing) => [DailyCarry, DailyOutput],
): ScanDailySequenceResult {
  return lax.scan(
    daily_step as unknown as (
      carry: JsTree<np.Array>,
      forcing: JsTree<np.Array>,
    ) => [JsTree<np.Array>, JsTree<np.Array>],
    daily_init as unknown as JsTree<np.Array>,
    daily_forcing as unknown as JsTree<np.Array>,
    { length: ndays },
  ) as unknown as ScanDailySequenceResult;
}

function makeScanDailySequence(
  ndays: number,
  daily_step: (carry: DailyCarry, forcing: DailyForcing) => [DailyCarry, DailyOutput],
): ScanDailySequenceFn {
  return (
    daily_init: DailyCarry,
    daily_forcing: DailyForcing,
  ): ScanDailySequenceResult => executeScanDailySequence(
    daily_init,
    daily_forcing,
    ndays,
    daily_step,
  );
}

function makeJittedScanDailySequence(
  ndays: number,
  daily_step: (carry: DailyCarry, forcing: DailyForcing) => [DailyCarry, DailyOutput],
): ScanDailySequenceFn {
  const runSequence = jit(((
    daily_init: DailyCarry,
    daily_forcing: DailyForcing,
  ): JsTree<np.Array> => executeScanDailySequence(
    daily_init,
    daily_forcing,
    ndays,
    daily_step,
  ) as unknown as JsTree<np.Array>) as unknown as (...args: unknown[]) => JsTree<np.Array>) as unknown as (
    daily_init: DailyCarry,
    daily_forcing: DailyForcing,
  ) => ScanDailySequenceResult;

  return (daily_init: DailyCarry, daily_forcing: DailyForcing): ScanDailySequenceResult => (
    runSequence(daily_init, daily_forcing)
  );
}

function runScanDailySequenceWithRunner(
  inputs: RunIntegrationInputs,
  daily_init: DailyCarry,
  runSequence: ScanDailySequenceFn,
): ScanDailySequenceResult {
  const daily_forcing = createTracedDailyForcing(inputs);

  const result = runSequence(daily_init, daily_forcing);

  const [final_carry] = result;

  // The scan owns its returned carry and outputs, but the large forcing arrays
  // and most initial carry leaves are internal-only and must be released here.
  disposeTracedDailyForcing(daily_forcing);
  disposeDailyBoundaryCarryIfNotAliased(daily_init, final_carry);

  return result;
}

function buildDailyBoundaryOutput<
  TCanopyState,
  TSoilState extends DailyBoundarySoilState<TValue>,
  TMetRolling,
  TFlag,
  TLaiPrev,
  TValue,
  TVector,
>(
  snapshot: DailyBoundarySnapshot<
    TCanopyState,
    TSoilState,
    TMetRolling,
    TFlag,
    TLaiPrev,
    TValue,
    TVector
  >,
): DailyBoundaryOutput<TValue, TVector> {
  return {
    gpp_avg: snapshot.gpp_avg,
    nee: snapshot.biology.nee,
    hetero_resp: snapshot.biology.hetero_resp,
    auto_resp: snapshot.biology.auto_resp,
    cleaf: snapshot.biology.cleaf,
    croot: snapshot.biology.croot,
    cstem: snapshot.biology.cstem,
    cgrain: snapshot.biology.cgrain,
    lai_alloc: snapshot.biology.lai_alloc,
    litter_cleaf: snapshot.biology.litter_cleaf,
    litter_croot: snapshot.biology.litter_croot,
    soc_total: snapshot.biology.soc_total,
    wliq: snapshot.sw_state.Wliq,
    psi: snapshot.sw_state.Psi,
    cstate: snapshot.biology.cstate,
    et_total: snapshot.et_total,
  };
}

function buildDailyBoundaryCarry<
  TCanopyState,
  TSoilState extends DailyBoundarySoilState<TValue>,
  TMetRolling,
  TFlag,
  TLaiPrev,
  TValue,
  TVector,
>(
  snapshot: DailyBoundarySnapshot<
    TCanopyState,
    TSoilState,
    TMetRolling,
    TFlag,
    TLaiPrev,
    TValue,
    TVector
  >,
): DailyBoundaryCarry<
  TCanopyState,
  TSoilState,
  TMetRolling,
  TFlag,
  TLaiPrev,
  TValue,
  TVector
> {
  return {
    cw_state: snapshot.cw_state,
    sw_state: snapshot.sw_state,
    met_rolling: snapshot.met_rolling,
    is_first_met: snapshot.is_first_met,
    cleaf: snapshot.biology.cleaf,
    croot: snapshot.biology.croot,
    cstem: snapshot.biology.cstem,
    cgrain: snapshot.biology.cgrain,
    litter_cleaf: snapshot.biology.litter_cleaf,
    litter_croot: snapshot.biology.litter_croot,
    compost: snapshot.biology.compost,
    soluble: snapshot.soluble,
    above: snapshot.biology.above,
    below: snapshot.biology.below,
    yield_: snapshot.biology.yield_,
    grain_fill: snapshot.biology.grain_fill,
    lai_alloc: snapshot.biology.lai_alloc,
    pheno: snapshot.biology.pheno,
    cstate: snapshot.biology.cstate,
    nstate: snapshot.biology.nstate,
    lai_prev: snapshot.lai_prev,
  };
}

function createInitialDailyBoundaryCarry<
  TCanopyState,
  TSoilState,
  TMetRolling,
  TFlag,
  TLaiPrev,
  TValue,
  TVector,
>(
  cw_state: TCanopyState,
  sw_state: TSoilState,
  met_rolling: TMetRolling,
  is_first_met: TFlag,
  cstate: TVector,
  nstate: TValue,
  lai_prev: TLaiPrev,
  createZero: () => TValue,
  createOne: () => TValue,
): DailyBoundaryCarry<
  TCanopyState,
  TSoilState,
  TMetRolling,
  TFlag,
  TLaiPrev,
  TValue,
  TVector
> {
  return {
    cw_state,
    sw_state,
    met_rolling,
    is_first_met,
    cleaf: createZero(),
    croot: createZero(),
    cstem: createZero(),
    cgrain: createZero(),
    litter_cleaf: createZero(),
    litter_croot: createZero(),
    compost: createZero(),
    soluble: createZero(),
    above: createZero(),
    below: createZero(),
    yield_: createZero(),
    grain_fill: createZero(),
    lai_alloc: createZero(),
    pheno: createOne(),
    cstate,
    nstate,
    lai_prev,
  };
}

function buildDailyBoundaryTransition<
  TCanopyState,
  TSoilState extends DailyBoundarySoilState<TValue>,
  TMetRolling,
  TFlag,
  TLaiPrev,
  TValue,
  TVector,
>(
  snapshot: DailyBoundarySnapshot<
    TCanopyState,
    TSoilState,
    TMetRolling,
    TFlag,
    TLaiPrev,
    TValue,
    TVector
  >,
): DailyBoundaryTransition<
  DailyBoundaryCarry<
    TCanopyState,
    TSoilState,
    TMetRolling,
    TFlag,
    TLaiPrev,
    TValue,
    TVector
  >,
  DailyBoundaryOutput<TValue, TVector>
> {
  return {
    next_carry: buildDailyBoundaryCarry(snapshot),
    output: buildDailyBoundaryOutput(snapshot),
  };
}

function createDailyBoundarySnapshot<
  TCanopyState,
  TSoilState extends DailyBoundarySoilState<TValue>,
  TMetRolling,
  TFlag,
  TLaiPrev,
  TValue,
  TVector,
>(
  carry_soluble: TValue,
  lai_prev: TLaiPrev,
  cw_state: TCanopyState,
  sw_state: TSoilState,
  met_rolling: TMetRolling,
  is_first_met: TFlag,
  et_acc: TValue,
  gpp_avg: TValue,
  biology: DailyBoundaryBiologyState<TValue, TVector>,
): DailyBoundarySnapshot<TCanopyState, TSoilState, TMetRolling, TFlag, TLaiPrev, TValue, TVector> {
  return {
    cw_state,
    sw_state,
    met_rolling,
    is_first_met,
    soluble: carry_soluble,
    lai_prev,
    gpp_avg,
    et_total: et_acc,
    biology,
  };
}

function buildScanDailyTransition(
  carry: DailyCarry,
  forcing: DailyForcing,
  final_hourly: HourlyCarry,
  biology_update: DailyBiologyUpdate,
  gpp_avg: np.Array,
): ScanDailyTransition {
  const snapshot = createDailyBoundarySnapshot(
    carry.soluble,
    forcing.lai,
    final_hourly.cw_state,
    final_hourly.sw_state,
    final_hourly.met_rolling,
    final_hourly.is_first_met,
    final_hourly.et_acc,
    gpp_avg,
    biology_update,
  );

  return buildDailyBoundaryTransition(snapshot);
}

function disposeDailyBoundaryCarryIfNotAliased(
  initial: DailyBoundaryCarry<
    CanopyWaterState,
    SoilWaterState,
    np.Array,
    np.Array,
    np.Array,
    np.Array,
    np.Array
  >,
  escaped: DailyBoundaryCarry<
    CanopyWaterState,
    SoilWaterState,
    np.Array,
    np.Array,
    np.Array,
    np.Array,
    np.Array
  >,
): void {
  const disposeIfNotAliased = (i: np.Array, e: np.Array) => {
    if (i !== e) i.dispose();
  };

  disposeIfNotAliased(initial.cw_state.CanopyStorage, escaped.cw_state.CanopyStorage);
  disposeIfNotAliased(initial.cw_state.SWE, escaped.cw_state.SWE);
  disposeIfNotAliased(initial.cw_state.swe_i, escaped.cw_state.swe_i);
  disposeIfNotAliased(initial.cw_state.swe_l, escaped.cw_state.swe_l);

  disposeIfNotAliased(initial.sw_state.WatSto, escaped.sw_state.WatSto);
  disposeIfNotAliased(initial.sw_state.PondSto, escaped.sw_state.PondSto);
  disposeIfNotAliased(initial.sw_state.MaxWatSto, escaped.sw_state.MaxWatSto);
  disposeIfNotAliased(initial.sw_state.MaxPondSto, escaped.sw_state.MaxPondSto);
  disposeIfNotAliased(initial.sw_state.FcSto, escaped.sw_state.FcSto);
  disposeIfNotAliased(initial.sw_state.Wliq, escaped.sw_state.Wliq);
  disposeIfNotAliased(initial.sw_state.Psi, escaped.sw_state.Psi);
  disposeIfNotAliased(initial.sw_state.Sat, escaped.sw_state.Sat);
  disposeIfNotAliased(initial.sw_state.Kh, escaped.sw_state.Kh);
  disposeIfNotAliased(initial.sw_state.beta, escaped.sw_state.beta);

  disposeIfNotAliased(initial.met_rolling, escaped.met_rolling);
  disposeIfNotAliased(initial.is_first_met, escaped.is_first_met);

  disposeIfNotAliased(initial.cleaf, escaped.cleaf);
  disposeIfNotAliased(initial.croot, escaped.croot);
  disposeIfNotAliased(initial.cstem, escaped.cstem);
  disposeIfNotAliased(initial.cgrain, escaped.cgrain);
  disposeIfNotAliased(initial.litter_cleaf, escaped.litter_cleaf);
  disposeIfNotAliased(initial.litter_croot, escaped.litter_croot);
  disposeIfNotAliased(initial.compost, escaped.compost);
  disposeIfNotAliased(initial.soluble, escaped.soluble);
  disposeIfNotAliased(initial.above, escaped.above);
  disposeIfNotAliased(initial.below, escaped.below);
  disposeIfNotAliased(initial.yield_, escaped.yield_);
  disposeIfNotAliased(initial.grain_fill, escaped.grain_fill);
  disposeIfNotAliased(initial.lai_alloc, escaped.lai_alloc);
  disposeIfNotAliased(initial.pheno, escaped.pheno);
  disposeIfNotAliased(initial.cstate, escaped.cstate);
  disposeIfNotAliased(initial.nstate, escaped.nstate);
  disposeIfNotAliased(initial.lai_prev, escaped.lai_prev);
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

function withSharedIntegrationResources<TResult>(
  inputs: RunIntegrationInputs,
  run: (context: IntegrationRunContext) => TResult,
): TResult {
  const solverKind = resolveSolverKind(inputs.phydro_optimizer);
  const ndays = leadingLength(inputs.daily_lai);
  const resources = createSharedIntegrationResources(inputs);

  try {
    return run({ solverKind, ndays, resources });
  } finally {
    disposeSharedIntegrationResources(resources);
  }
}

function runIntegrationWithRunnerFactory(
  inputs: RunIntegrationInputs,
  runnerFactory: ScanDailySequenceRunnerFactory,
): IntegrationResult {
  return withSharedIntegrationResources(inputs, (context) => {
    const executionResources = createScanDailyStepResources(
      inputs,
      context.resources,
      context.solverKind,
    );

    try {
      const initialState = consumeInitialScanDailyCarryFromResources(context.resources);
      return runScanDailySequenceWithRunner(
        inputs,
        initialState,
        runnerFactory(context.ndays, executionResources.daily_step),
      );
    } finally {
      disposeScanDailyStepResources(executionResources);
    }
  });
}

function createTracedDailyForcing(inputs: RunIntegrationInputs): DailyForcing {
  return {
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
}

function disposeTracedDailyForcing(forcing: DailyForcing): void {
  forcing.hourly_temp.dispose();
  forcing.hourly_rg.dispose();
  forcing.hourly_prec.dispose();
  forcing.hourly_vpd.dispose();
  forcing.hourly_pres.dispose();
  forcing.hourly_co2.dispose();
  forcing.hourly_wind.dispose();
  forcing.lai.dispose();
  forcing.management_type.dispose();
  forcing.management_c_in.dispose();
  forcing.management_c_out.dispose();
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
  )) as Disposable & {
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
  using _alloc_npp_day = alloc_result.nppDay;
  void _alloc_npp_day;
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
  )) as Disposable & { ctend: np.Array; ntend: np.Array };
  using cstate_plus_ctend = carry.cstate.add(decomp_result.ctend);
  const cstate = cstate_plus_ctend.add(input_cfract);
  const nstate = carry.nstate.add(decomp_result.ntend);

  using ctend_neg = decomp_result.ctend.neg();
  using hetero_resp_num = np.sum(ctend_neg);
  using hetero_resp_day = hetero_resp_num.div(24.0);
  const hetero_resp = hetero_resp_day.div(3600.0);
  using auto_resp_day = alloc_auto_resp.div(24.0);
  const auto_resp = auto_resp_day.div(3600.0);
  using total_resp = hetero_resp.add(auto_resp);
  const nee = total_resp.sub(gpp_avg);
  const soc_total = np.sum(cstate);

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
    cstate,
    nstate,
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
    // jax-js-lint: allow-ref
    WatSto: watSto.ref,
    PondSto: pondSto,
    // jax-js-lint: allow-ref
    MaxWatSto: maxWatSto.ref,
    // jax-js-lint: allow-ref
    MaxPondSto: maxpondArr.ref,
    // jax-js-lint: allow-ref
    FcSto: fcSto.ref,
    // jax-js-lint: allow-ref
    Wliq: wliq.ref,
    Psi: psi,
    // jax-js-lint: allow-ref
    Sat: sat.ref,
    Kh: kh,
    // jax-js-lint: allow-ref
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
    using gpp_zero = gpp_scaled.mul(0.0);
    using aj_zero = aj.mul(0.0);
    using gs_zero = gs.mul(0.0);
    using vcmax_zero = vcmax_hr.mul(0.0);
    const gpp_hr = np.where(has_light, gpp_scaled, gpp_zero);
    const aj_guarded = np.where(has_light, aj, aj_zero);
    const gs_guarded = np.where(has_light, gs, gs_zero);
    const vcmax_guarded = np.where(has_light, vcmax_hr, vcmax_zero);
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
    using tr_zero = tr_raw.mul(0.0);
    const tr_phydro = np.where(tr_valid, tr_raw, tr_zero);

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
    using use_initial_met = carry.is_first_met;
    const new_met_rolling = np.where(use_initial_met, met_daily, smoothed);
    tree.dispose(cw_flux);

    using gpp_valid_finite = np.isfinite(aj_guarded);
    using gpp_valid_nonneg = aj_guarded.greaterEqual(0.0);
    using gpp_valid = gpp_valid_finite.mul(gpp_valid_nonneg);
    using gpp_add_zero = gpp_hr.mul(0.0);
    using gpp_count_zero = gpp_valid.mul(0.0);
    using gpp_count_one = gpp_count_zero.add(1.0);
    const gpp_to_add = np.where(gpp_valid, gpp_hr, gpp_add_zero);
    using gpp_count_inc = np.where(gpp_valid, gpp_count_one, gpp_count_zero);
    const num_gpp = carry.num_gpp.add(gpp_count_inc);

    using vcmax_valid_finite = np.isfinite(vcmax_guarded);
    using vcmax_valid_positive = vcmax_guarded.greater(0.0);
    using vcmax_valid = vcmax_valid_finite.mul(vcmax_valid_positive);
    using vcmax_add_zero = vcmax_guarded.mul(0.0);
    using vcmax_count_zero = vcmax_valid.mul(0.0);
    using vcmax_count_one = vcmax_count_zero.add(1.0);
    const vcmax_to_add = np.where(vcmax_valid, vcmax_guarded, vcmax_add_zero);
    using vcmax_count_inc = np.where(vcmax_valid, vcmax_count_one, vcmax_count_zero);
    const num_vcmax = carry.num_vcmax.add(vcmax_count_inc);

    using new_met_temp = new_met_rolling.slice(0);
    using new_met_prec = new_met_rolling.slice(1);
    const new_temp_acc = carry.temp_acc.add(new_met_temp);
    using precip_increment = new_met_prec.mul(TIME_STEP * 3600.0);
    const new_precip_acc = carry.precip_acc.add(precip_increment);
    const new_gpp_acc = carry.gpp_acc.add(gpp_to_add);
    const new_vcmax_acc = carry.vcmax_acc.add(vcmax_to_add);
    const new_et_acc = carry.et_acc.add(et_hr);
    using _not_first_met_false = np.array(false, { dtype: np.bool });
    const not_first_met = np.where(use_initial_met, _not_first_met_false, use_initial_met);
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
  solverKind: SolverKind,
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
    using _gpp_valid = final_hourly.num_gpp.greater(0.0);
    using _gpp_zero = _gpp_ratio.mul(0.0);
    const gpp_avg = np.where(
      _gpp_valid,
      _gpp_ratio,
      _gpp_zero,
    );
    using _vcmax_ratio = final_hourly.vcmax_acc.div(final_hourly.num_vcmax);
    using _vcmax_valid = final_hourly.num_vcmax.greater(0.0);
    using _vcmax_zero = _vcmax_ratio.mul(0.0);
    const vcmax_avg = np.where(
      _vcmax_valid,
      _vcmax_ratio,
      _vcmax_zero,
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

export function runIntegration(inputs: RunIntegrationInputs): IntegrationResult {
  return runIntegrationWithRunnerFactory(inputs, makeScanDailySequence);
}

export function runIntegrationJit(inputs: RunIntegrationInputs): IntegrationResult {
  return runIntegrationWithRunnerFactory(inputs, makeJittedScanDailySequence);
}