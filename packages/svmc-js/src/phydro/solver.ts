import { grad, hessian, lax, tree } from "@hamk-uas/jax-js-nonconsuming";
import { np, retainArray } from "../precision.js";
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
export const DEFAULT_PHYDRO_OPTIMIZER = "projected_lbfgs";
export type PhydroOptimizer = "projected_lbfgs" | "projected_newton";

const LOG_JMAX_LO = -10.0;
const LOG_JMAX_HI = 10.0;
const DPSI_LO = 1e-4;
const DPSI_HI = 1e6;

const STARTS = [
  [4.0, 1.0],
  [1.0, 0.05],
  [LOG_JMAX_LO, DPSI_LO],
] as const;

const MAXITER = 16;
const STEP_SIZES = [
  1e-4, 3e-4, 1e-3, 3e-3,
  1e-2, 3e-2, 1e-1, 3e-1,
  1.0, 3.0, 10.0,
] as const;
const BOUND_TOL = 1e-10;
const HESS_REG = 1e-4;
const MEMORY = 4;
const CURV_EPS = 1e-12;
const POLISH_ITERS = 4;

const NEWTON_MAXITER = 12;
const NEWTON_STEP_SIZES = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0] as const;
const LM_INIT = 1e-2;
const LM_SHRINK = 0.5;
const LM_GROW = 4.0;
const LM_MIN = 1e-8;
const LM_MAX = 1e6;

type OptimResult = {
  jmax: np.Array;
  dpsi: np.Array;
  objectiveLoss: np.Array;
};

type LbfgsCarry = {
  x: np.Array;
  sHist: np.Array;
  yHist: np.Array;
  rhoHist: np.Array;
  validHist: np.Array;
};

type NewtonCarry = {
  x: np.Array;
  lam: np.Array;
};

function packParams(logJmax: np.Array, dpsi: np.Array): np.Array {
  return np.stack([logJmax, dpsi]);
}

function objective(
  params: np.Array,
  psiSoil: np.Array,
  parCost: ParCost,
  parPhotosynth: ParPhotosynth,
  parPlant: ParPlant,
  parEnv: ParEnv,
): np.Array {
  using logJmax = lax.dynamicIndexInDim(params, 0, 0, false);
  using dpsi = lax.dynamicIndexInDim(params, 1, 0, false);
  return fnProfit(
    logJmax,
    dpsi,
    psiSoil,
    parCost,
    parPhotosynth,
    parPlant,
    parEnv,
    "PM",
    true,
  );
}

function row(arr: np.Array, index: number): np.Array {
  return lax.dynamicIndexInDim(arr, index, 0, false);
}

function el(arr: np.Array, index: number): np.Array {
  return lax.dynamicIndexInDim(arr, index, 0, false);
}

function dot2(left: np.Array, right: np.Array): np.Array {
  using prod = left.mul(right);
  return np.sum(prod);
}

function squaredNorm(x: np.Array): np.Array {
  using sq = np.power(x, 2);
  return np.sum(sq);
}

function selectRow(stacked: np.Array, index: np.Array): np.Array {
  using idx = np.arange(stacked.shape[0]);
  using mask = idx.equal(index);
  using weights = mask.astype(stacked.dtype).reshape([stacked.shape[0], 1]);
  using weighted = stacked.mul(weights);
  return np.sum(weighted, 0);
}

function selectScalar(stacked: np.Array, index: np.Array): np.Array {
  using idx = np.arange(stacked.shape[0]);
  using mask = idx.equal(index);
  using weights = mask.astype(stacked.dtype);
  using weighted = stacked.mul(weights);
  return np.sum(weighted);
}

function evaluateCandidates(
  candidates: np.Array[],
  objAt: (x: np.Array) => np.Array,
): { x: np.Array; value: np.Array } {
  using candidateStack = np.stack(candidates);
  const safeValsList = candidates.map((candidate) => {
    using value = objAt(candidate);
    using finite = np.isfinite(value);
    return np.where(finite, value, np.inf);
  });
  using safeVals = np.stack(safeValsList);
  const bestIdx = np.argmin(safeVals);
  const bestX = selectRow(candidateStack, bestIdx);
  const bestVal = selectScalar(safeVals, bestIdx);
  bestIdx.dispose();
  for (const candidate of candidates) candidate.dispose();
  for (const value of safeValsList) value.dispose();
  return { x: bestX, value: bestVal };
}

function freeMask(x: np.Array, gradX: np.Array): np.Array {
  using x0 = el(x, 0);
  using x1 = el(x, 1);
  using g0 = el(gradX, 0);
  using g1 = el(gradX, 1);

  using atLo0 = np.abs(x0.sub(LOG_JMAX_LO)).lessEqual(BOUND_TOL);
  using atLo1 = np.abs(x1.sub(DPSI_LO)).lessEqual(BOUND_TOL);
  using atHi0 = np.abs(x0.sub(LOG_JMAX_HI)).lessEqual(BOUND_TOL);
  using atHi1 = np.abs(x1.sub(DPSI_HI)).lessEqual(BOUND_TOL);

  using activeLo0 = atLo0.mul(g0.greaterEqual(0.0));
  using activeLo1 = atLo1.mul(g1.greaterEqual(0.0));
  using activeHi0 = atHi0.mul(g0.lessEqual(0.0));
  using activeHi1 = atHi1.mul(g1.lessEqual(0.0));

  using free0Pre = np.where(activeLo0, 0.0, 1.0);
  using free0 = np.where(activeHi0, 0.0, free0Pre);
  using free1Pre = np.where(activeLo1, 0.0, 1.0);
  using free1 = np.where(activeHi1, 0.0, free1Pre);
  return np.stack([free0, free1]);
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
 * Projected Newton baseline with adaptive LM damping.
 *
 * Mirrors the selectable JAX fallback path.
 */
function projectedNewtonSolveImpl(
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
): np.Array {
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

  const objAt = (x: np.Array) => objective(
    x,
    psiSoil,
    parCost,
    parPhotosynth,
    parPlant,
    parEnv,
  );

  const step = (carry: NewtonCarry): NewtonCarry => {
    using fCur = objAt(carry.x);
    using gradX = grad(objAt)(carry.x);
    using hessX = hessian(objAt)(carry.x);
    using eye = np.eye(2, { dtype: carry.x.dtype });
    using lmEye = eye.mul(carry.lam);
    using hess = hessX.add(lmEye);
    using step = np.linalg.solve(hess, gradX);

    const candidates = NEWTON_STEP_SIZES.map((stepSize) => {
      using scaledStep = step.mul(stepSize);
      using candidateRaw = carry.x.sub(scaledStep);
      return projectParams(candidateRaw);
    });
    candidates.push(carry.x.ref);
    const best = evaluateCandidates(candidates, objAt);
    using threshold = fCur.sub(1e-14);
    using improved = best.value.less(threshold);
    using xNext = np.where(improved, best.x, carry.x);
    using shrunk = np.maximum(carry.lam.mul(LM_SHRINK), LM_MIN);
    using grown = np.minimum(carry.lam.mul(LM_GROW), LM_MAX);
    using lamNext = np.where(improved, shrunk, grown);
    best.x.dispose();
    best.value.dispose();
    return { x: xNext.ref, lam: lamNext.ref };
  };

  const initCarry: NewtonCarry = {
    x: np.array([4.0, 1.0]),
    lam: np.array(LM_INIT),
  };
  const scanBody = (carry: NewtonCarry, _i: np.Array): [NewtonCarry, np.Array] => {
    const next = step(carry);
    return [next, next.lam.ref];
  };
  using stepIds = np.arange(NEWTON_MAXITER);
  const [result, lmHistory] = lax.scan(
    scanBody as unknown as (
      carry: NewtonCarry,
      forcing: np.Array,
    ) => [NewtonCarry, np.Array],
    initCarry,
    stepIds,
    { length: NEWTON_MAXITER },
  ) as unknown as [NewtonCarry, np.Array];
  lmHistory.dispose();
  const finalX = result.x.ref;
  if (result.x !== initCarry.x) initCarry.x.dispose();
  if (result.lam !== initCarry.lam) initCarry.lam.dispose();
  tree.dispose(result);
  return finalX;
}

function lbfgsDirection(
  gradX: np.Array,
  sHist: np.Array,
  yHist: np.Array,
  rhoHist: np.Array,
  validHist: np.Array,
  free: np.Array,
): np.Array {
  using q0 = gradX.mul(free);
  const alphaRefs: np.Array[] = new Array(MEMORY);
  let q = q0.ref;

  for (let idx = MEMORY - 1; idx >= 0; idx -= 1) {
    using qPrev = q;
    using valid = el(validHist, idx);
    using sI = row(sHist, idx).mul(free);
    using yI = row(yHist, idx).mul(free);
    using rhoI = np.where(valid, el(rhoHist, idx), 0.0);
    using alphaRaw = rhoI.mul(dot2(sI, qPrev));
    const alphaI = np.where(valid, alphaRaw, 0.0);
    using alphaY = alphaI.mul(yI);
    using qNext = qPrev.sub(alphaY);
    q = qNext.ref;
    alphaRefs[idx] = alphaI.ref;
  }

  using valid0 = el(validHist, 0);
  using valid1 = el(validHist, 1);
  using valid2 = el(validHist, 2);
  using valid3 = el(validHist, 3);
  using s0 = row(sHist, 0).mul(free);
  using s1 = row(sHist, 1).mul(free);
  using s2 = row(sHist, 2).mul(free);
  using s3 = row(sHist, 3).mul(free);
  using y0 = row(yHist, 0).mul(free);
  using y1 = row(yHist, 1).mul(free);
  using y2 = row(yHist, 2).mul(free);
  using y3 = row(yHist, 3).mul(free);

  using yy0 = dot2(y0, y0);
  using yy1 = dot2(y1, y1);
  using yy2 = dot2(y2, y2);
  using yy3 = dot2(y3, y3);
  using sy0 = dot2(s0, y0);
  using sy1 = dot2(s1, y1);
  using sy2 = dot2(s2, y2);
  using sy3 = dot2(s3, y3);
  using use0 = valid0.mul(yy0.greater(CURV_EPS)).mul(sy0.greater(CURV_EPS));
  using use1 = valid1.mul(yy1.greater(CURV_EPS)).mul(sy1.greater(CURV_EPS));
  using use2 = valid2.mul(yy2.greater(CURV_EPS)).mul(sy2.greater(CURV_EPS));
  using use3 = valid3.mul(yy3.greater(CURV_EPS)).mul(sy3.greater(CURV_EPS));
  using gamma0 = sy0.div(yy0);
  using gamma1 = sy1.div(yy1);
  using gamma2 = sy2.div(yy2);
  using gamma3 = sy3.div(yy3);
  using gamma10 = np.where(use1, gamma1, gamma0);
  using gamma210 = np.where(use2, gamma2, gamma10);
  using gamma3210 = np.where(use3, gamma3, gamma210);
  const gamma = np.where(use0, gamma3210, np.where(use1, gamma3210, np.where(use2, gamma3210, np.where(use3, gamma3210, 1.0))));

  using r0 = q.mul(gamma);
  let r = r0.ref;
  for (let idx = 0; idx < MEMORY; idx += 1) {
    using rPrev = r;
    using valid = el(validHist, idx);
    using sI = row(sHist, idx).mul(free);
    using yI = row(yHist, idx).mul(free);
    using rhoI = np.where(valid, el(rhoHist, idx), 0.0);
    using beta = rhoI.mul(dot2(yI, rPrev));
    using alphaMinusBeta = alphaRefs[idx].sub(beta);
    using correction = alphaMinusBeta.mul(sI);
    using rNext = rPrev.add(correction);
    r = rNext.ref;
  }

  using direction = r.neg().mul(free);
  using fallback = q0.neg();
  using directionNormSq = squaredNorm(direction);
  using q0NormSq = squaredNorm(q0);
  using descentDot = dot2(direction, q0);
  using useFallback0 = directionNormSq.lessEqual(CURV_EPS * CURV_EPS);
  using useFallback1 = descentDot.greaterEqual(0.0);
  using useFallback2 = q0NormSq.lessEqual(CURV_EPS * CURV_EPS);
  using fallback01 = np.where(useFallback0, true, useFallback1);
  using useFallback = np.where(fallback01, true, useFallback2);
  const result = np.where(useFallback, fallback, direction);

  q.dispose();
  r.dispose();
  for (const alpha of alphaRefs) alpha.dispose();
  return result;
}

function updateMemory(
  sHist: np.Array,
  yHist: np.Array,
  rhoHist: np.Array,
  validHist: np.Array,
  step: np.Array,
  yVec: np.Array,
  free: np.Array,
  accept: np.Array,
): {
  sHist: np.Array;
  yHist: np.Array;
  rhoHist: np.Array;
  validHist: np.Array;
} {
  using stepFree = step.mul(free);
  using yFree = yVec.mul(free);
  using sy = dot2(stepFree, yFree);
  using acceptUpdate = accept.mul(np.isfinite(sy)).mul(sy.greater(CURV_EPS));
  using rho = np.where(acceptUpdate, np.array(1.0).div(sy), 0.0);

  using sShifted = np.stack([row(sHist, 1), row(sHist, 2), row(sHist, 3), stepFree]);
  using yShifted = np.stack([row(yHist, 1), row(yHist, 2), row(yHist, 3), yFree]);
  using rhoShifted = np.stack([el(rhoHist, 1), el(rhoHist, 2), el(rhoHist, 3), rho]);
  using validShifted = np.stack([el(validHist, 1), el(validHist, 2), el(validHist, 3), acceptUpdate]);

  const nextSHist = np.where(acceptUpdate, sShifted, sHist);
  const nextYHist = np.where(acceptUpdate, yShifted, yHist);
  const nextRhoHist = np.where(acceptUpdate, rhoShifted, rhoHist);
  const nextValidHist = np.where(acceptUpdate, validShifted, validHist);

  return {
    sHist: nextSHist,
    yHist: nextYHist,
    rhoHist: nextRhoHist,
    validHist: nextValidHist,
  };
}

function lbfgsSolveFromStartImpl(
  start: np.Array,
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
): np.Array {
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
  const objAt = (x: np.Array) => objective(
    x,
    psiSoil,
    parCost,
    parPhotosynth,
    parPlant,
    parEnv,
  );

  const initState: LbfgsCarry = {
    x: start.ref,
    sHist: np.zeros([MEMORY, 2]),
    yHist: np.zeros([MEMORY, 2]),
    rhoHist: np.zeros([MEMORY]),
    validHist: np.zeros([MEMORY], { dtype: np.bool }),
  };

  const step = (carry: LbfgsCarry): LbfgsCarry => {
    using fCur = objAt(carry.x);
    using gCur = grad(objAt)(carry.x);
    using free = freeMask(carry.x, gCur);
    using direction = lbfgsDirection(
      gCur,
      carry.sHist,
      carry.yHist,
      carry.rhoHist,
      carry.validHist,
      free,
    );

    const candidates = STEP_SIZES.map((stepSize) => {
      using scaledDir = direction.mul(stepSize);
      using candidateRaw = carry.x.add(scaledDir);
      return projectParams(candidateRaw);
    });
    candidates.push(carry.x.ref);
    const best = evaluateCandidates(candidates, objAt);
    using threshold = fCur.sub(1e-14);
    using improved = best.value.less(threshold);
    using xNext = np.where(improved, best.x, carry.x);
    using gNext = grad(objAt)(xNext);
    using stepVec = xNext.sub(carry.x);
    using yVec = gNext.sub(gCur);
    const updated = updateMemory(
      carry.sHist,
      carry.yHist,
      carry.rhoHist,
      carry.validHist,
      stepVec,
      yVec,
      free,
      improved,
    );

    best.x.dispose();
    best.value.dispose();
    return {
      x: xNext.ref,
      sHist: updated.sHist,
      yHist: updated.yHist,
      rhoHist: updated.rhoHist,
      validHist: updated.validHist,
    };
  };

  const scanBody = (carry: LbfgsCarry, i: np.Array): [LbfgsCarry, np.Array] => {
    const next = step(carry);
    using marker = np.array(0.0, { dtype: next.x.dtype });
    return [next, marker.ref];
  };
  using stepIds = np.arange(MAXITER);
  const [result, xHistory] = lax.scan(
    scanBody as unknown as (
      carry: LbfgsCarry,
      forcing: np.Array,
    ) => [LbfgsCarry, np.Array],
    initState,
    stepIds,
    { length: MAXITER },
  ) as unknown as [LbfgsCarry, np.Array];
  xHistory.dispose();
  const finalX = result.x.ref;
  if (result.x !== initState.x) initState.x.dispose();
  if (result.sHist !== initState.sHist) initState.sHist.dispose();
  if (result.yHist !== initState.yHist) initState.yHist.dispose();
  if (result.rhoHist !== initState.rhoHist) initState.rhoHist.dispose();
  if (result.validHist !== initState.validHist) initState.validHist.dispose();
  tree.dispose(result);
  return finalX;
}

function projectedNewtonPolishImpl(
  start: np.Array,
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
): np.Array {
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
  const objAt = (x: np.Array) => objective(
    x,
    psiSoil,
    parCost,
    parPhotosynth,
    parPlant,
    parEnv,
  );

  const step = (x: np.Array): np.Array => {
    using fCur = objAt(x);
    using gCur = grad(objAt)(x);
    using free = freeMask(x, gCur);
    using hessX = hessian(objAt)(x);
    using eye = np.eye(2, { dtype: x.dtype });
    using freeCol = free.reshape([2, 1]);
    using freeRow = free.reshape([1, 2]);
    using maskedBase = hessX.mul(freeCol).mul(freeRow);
    using invFree = np.array(1.0).sub(free);

    using diagAdd = eye.mul(invFree.reshape([2, 1]));
    using maskedHess = maskedBase.add(diagAdd).add(eye.mul(HESS_REG));
    using maskedGrad = gCur.mul(free);
    using newtonDirRaw = np.linalg.solve(maskedHess, maskedGrad).neg();
    using newtonDir = newtonDirRaw.mul(free);
    using gradDir = maskedGrad.neg();
    using newtonFinite = np.all(np.isfinite(newtonDir));
    using newtonDescent = dot2(newtonDir, maskedGrad).less(0.0);
    using newtonOk = newtonFinite.mul(newtonDescent);
    using direction = np.where(newtonOk, newtonDir, gradDir);

    const candidates = STEP_SIZES.map((stepSize) => {
      using scaledDir = direction.mul(stepSize);
      using candidateRaw = x.add(scaledDir);
      return projectParams(candidateRaw);
    });
    candidates.push(x.ref);
    const best = evaluateCandidates(candidates, objAt);
    using threshold = fCur.sub(1e-14);
    using improved = best.value.less(threshold);
    using xNext = np.where(improved, best.x, x);
    best.x.dispose();
    best.value.dispose();
    return xNext.ref;
  };

  const scanBody = (x: np.Array, _i: np.Array): [np.Array, np.Array] => {
    const next = step(x);
    using marker = np.array(0.0, { dtype: next.dtype });
    return [next, marker.ref];
  };
  using stepIds = np.arange(POLISH_ITERS);
  const [result, path] = lax.scan(
    scanBody as unknown as (
      carry: np.Array,
      forcing: np.Array,
    ) => [np.Array, np.Array],
    start,
    stepIds,
    { length: POLISH_ITERS },
  ) as unknown as [np.Array, np.Array];
  path.dispose();
  const finalX = result.ref;
  result.dispose();
  return finalX;
}

function lbfgsSolveImpl(
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
): np.Array {
  const polishedStarts = STARTS.map(([logJmax0, dpsi0]) => {
    using start = np.array([logJmax0, dpsi0]);
    const lbfgsX = lbfgsSolveFromStartImpl(
      start,
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
    const polished = projectedNewtonPolishImpl(
      lbfgsX,
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
    lbfgsX.dispose();
    return polished;
  });

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
  const objAt = (x: np.Array) => objective(
    x,
    psiSoil,
    parCost,
    parPhotosynth,
    parPlant,
    parEnv,
  );

  const safeVals = polishedStarts.map((candidate) => {
    using value = objAt(candidate);
    using finite = np.isfinite(value);
    return np.where(finite, value, np.inf);
  });

  let bestX = polishedStarts[0].ref;
  let bestVal = safeVals[0].ref;
  for (let idx = 1; idx < polishedStarts.length; idx += 1) {
    using better = safeVals[idx].less(bestVal);
    using nextX = np.where(better, polishedStarts[idx], bestX);
    using nextVal = np.where(better, safeVals[idx], bestVal);
    bestX.dispose();
    bestVal.dispose();
    bestX = nextX.ref;
    bestVal = nextVal.ref;
  }

  for (const candidate of polishedStarts) candidate.dispose();
  for (const value of safeVals) value.dispose();
  bestVal.dispose();
  return bestX;
}

function optimiseMidtermMultiLbfgsFlat(
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
): np.Array {
  const result = lbfgsSolveImpl(
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
  return result;
}

function optimiseMidtermMultiNewtonFlat(
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
): np.Array {
  const result = projectedNewtonSolveImpl(
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
  return result;
}

export function optimiseMidtermMulti(
  psiSoil: np.Array,
  parCost: ParCost,
  parPhotosynth: ParPhotosynth,
  parPlant: ParPlant,
  parEnv: ParEnv,
  solverKind: PhydroOptimizer = DEFAULT_PHYDRO_OPTIMIZER,
): OptimResult {
  const params = solverKind === "projected_newton"
    ? optimiseMidtermMultiNewtonFlat(
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
    )
    : optimiseMidtermMultiLbfgsFlat(
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
  using logJmax = el(params, 0);
  using dpsiRaw = el(params, 1);
  using jmaxRaw = np.exp(logJmax);
  const jmax = jmaxRaw.ref;
  const dpsi = dpsiRaw.ref;
  using objectiveLossRaw = objective(
    params,
    psiSoil,
    parCost,
    parPhotosynth,
    parPlant,
    parEnv,
  );
  const objectiveLoss = objectiveLossRaw.ref;
  params.dispose();
  return { jmax, dpsi, objectiveLoss };
}

/**
 * Full P-Hydro solver: compute optimal photosynthesis-hydraulics state.
 *
 * Port of `pmodel_hydraulics_numerical` from phydro_mod.f90.
 * Supports the JAX solver choices `projected_lbfgs` and `projected_newton`.
 *
 * All returned np.Array values must be disposed by the caller.
 */
export function pmodelHydraulicsNumerical(
  tc: np.ArrayLike,
  ppfd: np.ArrayLike,
  vpd: np.ArrayLike,
  co2: np.ArrayLike,
  sp: np.ArrayLike,
  fapar: np.ArrayLike,
  psiSoilVal: np.ArrayLike,
  rdarkLeaf: np.ArrayLike,
  conductivityVal: np.ArrayLike = 4e-16,
  psi50Val: np.ArrayLike = -3.46,
  bParam: np.ArrayLike = 2.0,
  alphaVal: np.ArrayLike = 0.1,
  gammaCost: np.ArrayLike = 0.5,
  kphio: np.ArrayLike = KPHIO,
  solverKind: PhydroOptimizer = DEFAULT_PHYDRO_OPTIMIZER,
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
  using psiSoilNp = retainArray(psiSoilVal);

  // Optimise
  const opt = (() => {
    using conductivityOpt = retainArray(conductivityVal);
    using psi50Opt = retainArray(psi50Val);
    using bOpt = retainArray(bParam);
    using alphaOpt = retainArray(alphaVal);
    using gammaOpt = retainArray(gammaCost);
    const parPlantOpt: ParPlant = {
      conductivity: conductivityOpt,
      psi50: psi50Opt,
      b: bOpt,
    };
    const parCostOpt: ParCost = {
      alpha: alphaOpt,
      gamma: gammaOpt,
    };

    using optTcArr = retainArray(tc);
    using optSpArr = retainArray(sp);
    const optKmm = calcKmm(optTcArr, optSpArr);
    const optGsStar = computeGammastar(optTcArr, optSpArr);
    using optKphioVal = retainArray(kphio);
    using optFtk = ftempKphio(optTcArr, false);
    const optPhi0 = optKphioVal.mul(optFtk);
    using optPpfdArr = retainArray(ppfd);
    using optFaparArr = retainArray(fapar);
    const optIabs = optPpfdArr.mul(optFaparArr);
    using optCo2Arr = retainArray(co2);
    using optSpTmpBase = retainArray(sp);
    using optSpTmp = optSpTmpBase.mul(1e-6);
    const optCa = optCo2Arr.mul(optSpTmp);
    using optPatm = retainArray(sp);
    using optDelta = retainArray(rdarkLeaf);

    const optViscWater = viscosityH2o(optTcArr, optSpArr);
    const optDensWater = densityH2o(optTcArr, optSpArr);
    using optEnvPatm = retainArray(sp);
    using optEnvTc = retainArray(tc);
    using optEnvVpd = retainArray(vpd);

    const result = optimiseMidtermMulti(
      psiSoilNp,
      parCostOpt,
      {
        kmm: optKmm,
        gammastar: optGsStar,
        phi0: optPhi0,
        Iabs: optIabs,
        ca: optCa,
        patm: optPatm,
        delta: optDelta,
      },
      parPlantOpt,
      {
        viscosityWater: optViscWater,
        densityWater: optDensWater,
        patm: optEnvPatm,
        tc: optEnvTc,
        vpd: optEnvVpd,
      },
      solverKind,
    );

    // Dispose locally-owned intermediates. Retained inputs above use `using`
    // so they balance both numeric callers and borrowed Array inputs safely.
    optKmm.dispose();
    optGsStar.dispose();
    optPhi0.dispose();
    optIabs.dispose();
    optCa.dispose();
    optViscWater.dispose();
    optDensWater.dispose();

    return result;
  })();

  using conductivityArr = retainArray(conductivityVal);
  using psi50Arr = retainArray(psi50Val);
  using bArr = retainArray(bParam);
  const parPlant: ParPlant = {
    conductivity: conductivityArr,
    psi50: psi50Arr,
    b: bArr,
  };

  using tcArr = retainArray(tc);
  using spArr = retainArray(sp);
  const kmm = calcKmm(tcArr, spArr);
  const gs_star = computeGammastar(tcArr, spArr);
  using kphioVal = retainArray(kphio);
  using ftk = ftempKphio(tcArr, false);
  const phi0 = kphioVal.mul(ftk);
  using ppfdArr = retainArray(ppfd);
  using faparArr = retainArray(fapar);
  const Iabs = ppfdArr.mul(faparArr);
  using co2Arr = retainArray(co2);
  using _spTmpBase = retainArray(sp);
  using _spTmp = _spTmpBase.mul(1e-6);
  const ca = co2Arr.mul(_spTmp);

  const viscWater = viscosityH2o(tcArr, spArr);
  const densWater = densityH2o(tcArr, spArr);

  using photosynthPatm = retainArray(sp);
  using deltaArr = retainArray(rdarkLeaf);
  const parPhotosynth: ParPhotosynth = {
    kmm,
    gammastar: gs_star,
    phi0,
    Iabs,
    ca,
    patm: photosynthPatm,
    delta: deltaArr,
  };
  using envPatm = retainArray(sp);
  using envTc = retainArray(tc);
  using envVpd = retainArray(vpd);
  const parEnv: ParEnv = {
    viscosityWater: viscWater,
    densityWater: densWater,
    patm: envPatm,
    tc: envTc,
    vpd: envVpd,
  };

  // Evaluate diagnostics at optimum
  using profitRaw = opt.objectiveLoss.neg();
  const profit = profitRaw.ref;
  const jmax = opt.jmax.ref;
  const dpsiOut = opt.dpsi.ref;
  tree.dispose(opt);
  using gsVal = calcGs(dpsiOut, psiSoilNp, parPlant, parEnv);
  const gs = gsVal.ref;

  const { ci: ciRaw, aj: ajRaw } = calcAssimLightLimited(gs, jmax, parPhotosynth);
  const ci = ciRaw.ref;
  const aj = ajRaw.ref;
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

  // Manually dispose owned intermediates; retained input handles above are
  // cleaned up by their `using` scopes.
  kmm.dispose();
  gs_star.dispose();
  phi0.dispose();
  Iabs.dispose();
  ca.dispose();
  viscWater.dispose();
  densWater.dispose();

  return { jmax, dpsi: dpsiOut, gs, aj, ci, chi, vcmax, profit, chiJmaxLim };
}
