import { clearCaches, jit, lax, tree, valueAndGrad } from "@hamk-uas/jax-js-nonconsuming";
import { np } from "../precision.js";
import {
  adam,
  applyUpdates,
  chain,
  clipByGlobalNorm,
  type OptState,
} from "@hamk-uas/jax-js-nonconsuming/optax";
import { fnProfit } from "./fn-profit.js";
import { calcGs } from "./calc-gs.js";
import { calcAssimLightLimited } from "./calc-assim-light-limited.js";
import { calcKmm } from "./calc-kmm.js";
import { gammastar as computeGammastar } from "./gammastar.js";
import { ftempKphio } from "./ftemp-kphio.js";
import { densityH2o } from "./density-h2o.js";
import { viscosityH2o } from "./viscosity-h2o.js";
import type {
  ParPlant,
  ParEnv,
  ParCost,
  ParPhotosynth,
} from "./scale-conductivity.js";

/**
 * Fortran default quantum yield efficiency from readvegpara_mod.
 */
const KPHIO = 0.087182;
const MAX_OPT_STEPS = 512;
const OPT_LEARNING_RATE = 0.05;
const OPT_CLIP_NORM = 10.0;
const OPTIMIZER = chain(
  clipByGlobalNorm(OPT_CLIP_NORM),
  adam(OPT_LEARNING_RATE),
);

type OptimResult = {
  jmax: np.Array;
  dpsi: np.Array;
  objectiveLoss: np.Array;
};

type OptimCarry = {
  params: np.Array;
  optState: OptState;
  bestParams: np.Array;
  bestLoss: np.Array;
};

function packParams(logJmax: np.Array, dpsi: np.Array): np.Array {
  return np.stack([logJmax, dpsi]);
}

/**
 * Project optimizer parameters back into the Fortran box bounds.
 */
function projectParams(params: np.Array): np.Array {
  using logJmax = lax.dynamicIndexInDim(params, 0, 0, false);
  using dpsi = lax.dynamicIndexInDim(params, 1, 0, false);
  using clippedLogJmax = np.clip(logJmax, -10.0, 10.0);
  using clippedDpsi = np.clip(dpsi, 1e-4, 1e6);
  return packParams(clippedLogJmax, clippedDpsi);
}

/**
 * Projected Optax solver for the 2D profit minimization objective.
 *
 * Uses traced autodiff via `valueAndGrad` and runs entirely in jax-js array
 * code so it remains safe to compose into larger `jit()`-compiled loops.
 */
function optimiseMidtermMultiImpl(
  psiSoil: np.Array,
  alpha: np.Array,
  gamma: np.Array,
  kmm: np.Array,
  gammastar: np.Array,
  phi0: np.Array,
  Iabs: np.Array,
  ca: np.Array,
  photosynthPatm: np.Array,
  delta: np.Array,
  conductivity: np.Array,
  psi50: np.Array,
  b: np.Array,
  viscosityWater: np.Array,
  densityWater: np.Array,
  envPatm: np.Array,
  tc: np.Array,
  vpd: np.Array,
): OptimResult {
  const parCost: ParCost = { alpha, gamma };
  const parPhotosynth: ParPhotosynth = {
    kmm,
    gammastar,
    phi0,
    Iabs,
    ca,
    patm: photosynthPatm,
    delta,
  };
  const parPlant: ParPlant = { conductivity, psi50, b };
  const parEnv: ParEnv = {
    viscosityWater,
    densityWater,
    patm: envPatm,
    tc,
    vpd,
  };

  const objective = (
    params: np.Array,
    psiSoilArg: np.Array,
    parCostArg: ParCost,
    parPhotosynthArg: ParPhotosynth,
    parPlantArg: ParPlant,
    parEnvArg: ParEnv,
  ) => {
    using logJmax = lax.dynamicIndexInDim(params, 0, 0, false);
    using dpsi = lax.dynamicIndexInDim(params, 1, 0, false);
    return fnProfit(
      logJmax,
      dpsi,
      psiSoilArg,
      parCostArg,
      parPhotosynthArg,
      parPlantArg,
      parEnvArg,
      "PM",
      true,
    );
  };
  using initParams = np.array([4.0, 1.0]);
  const initCarry = {
    params: initParams,
    optState: OPTIMIZER.init(initParams),
    bestParams: np.array([4.0, 1.0]),
    bestLoss: np.array(Infinity),
  };

  const step = (carry: OptimCarry, _: null): [OptimCarry, null] => {
    void _;
    const [loss, grads] = valueAndGrad(objective, { argnums: 0 })(
      carry.params,
      psiSoil,
      parCost,
      parPhotosynth,
      parPlant,
      parEnv,
    );
    const better = loss.less(carry.bestLoss);
    const bestParams = np.where(better, carry.params, carry.bestParams);
    const bestLoss = np.where(better, loss, carry.bestLoss);
    const [updates, optState] = OPTIMIZER.update(grads, carry.optState, carry.params);
    const params = projectParams(applyUpdates(carry.params, updates));
    return [{ params, optState, bestParams, bestLoss }, null];
  };

  const [finalCarry] = lax.scan(step, initCarry, null, { length: MAX_OPT_STEPS });

  using finalLoss = objective(
    finalCarry.params,
    psiSoil,
    parCost,
    parPhotosynth,
    parPlant,
    parEnv,
  );
  using better = finalLoss.less(finalCarry.bestLoss);
  using bestParams = np.where(better, finalCarry.params, finalCarry.bestParams);
  using bestLoss = np.where(better, finalLoss, finalCarry.bestLoss);
  using logJmax = lax.dynamicIndexInDim(bestParams, 0, 0, false);
  using dpsiRaw = lax.dynamicIndexInDim(bestParams, 1, 0, false);
  using jmaxRaw = np.exp(logJmax);
  const jmax = jmaxRaw.add(0);
  const dpsi = dpsiRaw.add(0);
  const objectiveLoss = bestLoss.add(0);
  tree.dispose(finalCarry);
  return {
    jmax,
    dpsi,
    objectiveLoss,
  };
}

function optimiseMidtermMultiFlat(
  psiSoil: np.Array,
  alpha: np.Array,
  gamma: np.Array,
  kmm: np.Array,
  gammastar: np.Array,
  phi0: np.Array,
  Iabs: np.Array,
  ca: np.Array,
  photosynthPatm: np.Array,
  delta: np.Array,
  conductivity: np.Array,
  psi50: np.Array,
  b: np.Array,
  viscosityWater: np.Array,
  densityWater: np.Array,
  envPatm: np.Array,
  tc: np.Array,
  vpd: np.Array,
): OptimResult {
  using core = jit(optimiseMidtermMultiImpl);
  const result = core(
    psiSoil,
    alpha,
    gamma,
    kmm,
    gammastar,
    phi0,
    Iabs,
    ca,
    photosynthPatm,
    delta,
    conductivity,
    psi50,
    b,
    viscosityWater,
    densityWater,
    envPatm,
    tc,
    vpd,
  );
  clearCaches();
  return result;
}

export function optimiseMidtermMulti(
  psiSoil: np.Array,
  parCost: ParCost,
  parPhotosynth: ParPhotosynth,
  parPlant: ParPlant,
  parEnv: ParEnv,
): OptimResult {
  return optimiseMidtermMultiFlat(
    psiSoil,
    parCost.alpha,
    parCost.gamma,
    parPhotosynth.kmm,
    parPhotosynth.gammastar,
    parPhotosynth.phi0,
    parPhotosynth.Iabs,
    parPhotosynth.ca,
    parPhotosynth.patm,
    parPhotosynth.delta,
    parPlant.conductivity,
    parPlant.psi50,
    parPlant.b,
    parEnv.viscosityWater,
    parEnv.densityWater,
    parEnv.patm,
    parEnv.tc,
    parEnv.vpd,
  );
}

/**
 * Full P-Hydro solver: compute optimal photosynthesis-hydraulics state.
 *
 * Port of `pmodel_hydraulics_numerical` from phydro_mod.f90.
 * Uses projected Optax Adam updates inside traced jax-js code.
 *
 * All returned np.Array values must be disposed by the caller.
 */
export function pmodelHydraulicsNumerical(
  tc: number,
  ppfd: number,
  vpd: number,
  co2: number,
  sp: number,
  fapar: number,
  psiSoilVal: number,
  rdarkLeaf: number,
  conductivityVal = 4e-16,
  psi50Val = -3.46,
  bParam = 2.0,
  alphaVal = 0.1,
  gammaCost = 0.5,
  kphio = KPHIO,
): {
  jmax: np.Array;
  dpsi: np.Array;
  gs: np.Array;
  aj: np.Array;
  ci: np.Array;
  chi: np.Array;
  vcmax: np.Array;
  profit: np.Array;
  chiJmaxLim: np.Array;
} {
  using psiSoilNp = np.array(psiSoilVal);

  // Optimise
  const opt = (() => {
    const parPlantOpt: ParPlant = {
      conductivity: np.array(conductivityVal),
      psi50: np.array(psi50Val),
      b: np.array(bParam),
    };
    const parCostOpt: ParCost = {
      alpha: np.array(alphaVal),
      gamma: np.array(gammaCost),
    };

    using optTcArr = np.array(tc);
    using optSpArr = np.array(sp);
    const optKmm = calcKmm(optTcArr, optSpArr);
    const optGsStar = computeGammastar(optTcArr, optSpArr);
    using optKphioVal = np.array(kphio);
    using optFtk = ftempKphio(optTcArr, false);
    const optPhi0 = optKphioVal.mul(optFtk);
    using optPpfdArr = np.array(ppfd);
    using optFaparArr = np.array(fapar);
    const optIabs = optPpfdArr.mul(optFaparArr);
    using optCo2Arr = np.array(co2);
    using optSpTmp = np.array(sp * 1e-6);
    const optCa = optCo2Arr.mul(optSpTmp);
    const optPatm = np.array(sp);
    const optDelta = np.array(rdarkLeaf);

    const optViscWater = viscosityH2o(optTcArr, optSpArr);
    const optDensWater = densityH2o(optTcArr, optSpArr);
    const optEnvPatm = np.array(sp);
    const optEnvTc = np.array(tc);
    const optEnvVpd = np.array(vpd);

    const result = optimiseMidtermMultiFlat(
      psiSoilNp,
      parCostOpt.alpha,
      parCostOpt.gamma,
      optKmm,
      optGsStar,
      optPhi0,
      optIabs,
      optCa,
      optPatm,
      optDelta,
      parPlantOpt.conductivity,
      parPlantOpt.psi50,
      parPlantOpt.b,
      optViscWater,
      optDensWater,
      optEnvPatm,
      optEnvTc,
      optEnvVpd,
    );

    parPlantOpt.conductivity.dispose();
    parPlantOpt.psi50.dispose();
    parPlantOpt.b.dispose();
    parCostOpt.alpha.dispose();
    parCostOpt.gamma.dispose();
    optKmm.dispose();
    optGsStar.dispose();
    optPhi0.dispose();
    optIabs.dispose();
    optCa.dispose();
    optPatm.dispose();
    optDelta.dispose();
    optViscWater.dispose();
    optDensWater.dispose();
    optEnvPatm.dispose();
    optEnvTc.dispose();
    optEnvVpd.dispose();

    return result;
  })();

  using parPlant = tree.makeDisposable({
    conductivity: np.array(conductivityVal),
    psi50: np.array(psi50Val),
    b: np.array(bParam),
  }) as ParPlant;

  using tcArr = np.array(tc);
  using spArr = np.array(sp);
  const kmm = calcKmm(tcArr, spArr);
  const gs_star = computeGammastar(tcArr, spArr);
  using kphioVal = np.array(kphio);
  using ftk = ftempKphio(tcArr, false);
  const phi0 = kphioVal.mul(ftk);
  using ppfdArr = np.array(ppfd);
  using faparArr = np.array(fapar);
  const Iabs = ppfdArr.mul(faparArr);
  using co2Arr = np.array(co2);
  using _spTmp = np.array(sp * 1e-6);
  const ca = co2Arr.mul(_spTmp);
  const parPhotosynthPatm = np.array(sp);
  const parPhotosynthDelta = np.array(rdarkLeaf);

  const viscWater = viscosityH2o(tcArr, spArr);
  const densWater = densityH2o(tcArr, spArr);
  const parEnvPatm = np.array(sp);
  const parEnvTc = np.array(tc);
  const parEnvVpd = np.array(vpd);

  using parPhotosynth = tree.makeDisposable({
    kmm,
    gammastar: gs_star,
    phi0,
    Iabs,
    ca,
    patm: parPhotosynthPatm,
    delta: parPhotosynthDelta,
  }) as ParPhotosynth;
  using parEnv = tree.makeDisposable({
    viscosityWater: viscWater,
    densityWater: densWater,
    patm: parEnvPatm,
    tc: parEnvTc,
    vpd: parEnvVpd,
  }) as ParEnv;

  // Evaluate diagnostics at optimum
  using profitRaw = opt.objectiveLoss.neg();
  const profit = np.array(profitRaw.item() as number);
  const jmax = opt.jmax.add(0);
  const dpsiOut = opt.dpsi.add(0);
  tree.dispose(opt);
  using gsVal = calcGs(dpsiOut, psiSoilNp, parPlant, parEnv);
  const gs = np.array(gsVal.item() as number); // clone to decouple lifetime

  const { ci: ciRaw, aj: ajRaw } = calcAssimLightLimited(gs, jmax, parPhotosynth);
  const ci = np.array(ciRaw.item() as number);
  const aj = np.array(ajRaw.item() as number);
  ciRaw.dispose();
  ajRaw.dispose();

  // vcmax = aj * (ci + kmm) / (ci*(1-delta) - (gammastar + kmm*delta))
  using _ciKmm = ci.add(parPhotosynth.kmm);
  using _ajCiKmm = aj.mul(_ciKmm);
  using _oneMinusDelta = np.array(1).sub(parPhotosynth.delta);
  using _ciDenom1 = ci.mul(_oneMinusDelta);
  using _kmmDelta = parPhotosynth.kmm.mul(parPhotosynth.delta);
  using _gsKmmDelta = parPhotosynth.gammastar.add(_kmmDelta);
  using _denom = _ciDenom1.sub(_gsKmmDelta);
  const vcmax = _ajCiKmm.div(_denom);

  const chi = ci.div(parPhotosynth.ca);
  const chiJmaxLim = np.array(0);

  return { jmax, dpsi: dpsiOut, gs, aj, ci, chi, vcmax, profit, chiJmaxLim };
}
