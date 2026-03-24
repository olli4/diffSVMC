import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";
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

/**
 * Project a value into [lo, hi].
 */
function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

/**
 * Evaluate the negated profit function at [logJmax, dpsi] (for minimisation).
 * Caller must dispose the returned np.Array.
 */
function evalObjective(
  logJmax: number,
  dpsi: number,
  psiSoil: np.Array,
  parCost: ParCost,
  parPhotosynth: ParPhotosynth,
  parPlant: ParPlant,
  parEnv: ParEnv,
): number {
  using lj = np.array(logJmax);
  using dp = np.array(dpsi);
  using result = fnProfit(
    lj,
    dp,
    psiSoil,
    parCost,
    parPhotosynth,
    parPlant,
    parEnv,
    "PM",
    true, // doOptim = negate for minimisation
  );
  return result.item() as number;
}

/**
 * Finite-difference gradient of the objective, matching Fortran h=0.001.
 */
function fdGradient(
  x: [number, number],
  psiSoil: np.Array,
  parCost: ParCost,
  parPhotosynth: ParPhotosynth,
  parPlant: ParPlant,
  parEnv: ParEnv,
): { f: number; g: [number, number] } {
  const h = 0.001;
  const f0 = evalObjective(x[0], x[1], psiSoil, parCost, parPhotosynth, parPlant, parEnv);
  const f1 = evalObjective(x[0] + h, x[1], psiSoil, parCost, parPhotosynth, parPlant, parEnv);
  const f2 = evalObjective(x[0], x[1] + h, psiSoil, parCost, parPhotosynth, parPlant, parEnv);
  return {
    f: f0,
    g: [(f1 - f0) / h, (f2 - f0) / h],
  };
}

/** Bounds for the 2D L-BFGS-B problem. */
const BOUNDS: [[number, number], [number, number]] = [
  [-10, 10],
  [1e-4, 1e6],
];

/**
 * Projected L-BFGS optimisation for 2D box-constrained minimisation.
 *
 * Replaces Fortran's L-BFGS-B (setulb) with finite-difference gradients.
 * Matches Fortran settings: factr=1e7, pgtol=1e-5, maxIter=1000.
 */
export function optimiseMidtermMulti(
  psiSoil: np.Array,
  parCost: ParCost,
  parPhotosynth: ParPhotosynth,
  parPlant: ParPlant,
  parEnv: ParEnv,
): { logJmax: number; dpsi: number } {
  // Initial guess matching Fortran
  let x: [number, number] = [4.0, 1.0];

  // L-BFGS memory
  const m = 5;
  const sHistory: [number, number][] = [];
  const yHistory: [number, number][] = [];
  const rhoHistory: number[] = [];

  const maxIter = 1000;
  const ftol = 1e7 * 2.220446049250313e-16; // factr * machine_eps
  const gtol = 1e-5;

  let { f: fPrev, g: gPrev } = fdGradient(x, psiSoil, parCost, parPhotosynth, parPlant, parEnv);

  for (let iter = 0; iter < maxIter; iter++) {
    // Check gradient convergence
    const gInf = Math.max(Math.abs(gPrev[0]), Math.abs(gPrev[1]));
    if (gInf < gtol) break;

    // L-BFGS two-loop recursion to compute search direction
    const q: [number, number] = [gPrev[0], gPrev[1]];
    const alphas: number[] = [];
    for (let i = sHistory.length - 1; i >= 0; i--) {
      const ai =
        rhoHistory[i] * (sHistory[i][0] * q[0] + sHistory[i][1] * q[1]);
      alphas.unshift(ai);
      q[0] -= ai * yHistory[i][0];
      q[1] -= ai * yHistory[i][1];
    }

    // Initial Hessian approximation (scalar)
    let H0 = 1.0;
    if (sHistory.length > 0) {
      const lastIdx = sHistory.length - 1;
      const sy =
        sHistory[lastIdx][0] * yHistory[lastIdx][0] +
        sHistory[lastIdx][1] * yHistory[lastIdx][1];
      const yy =
        yHistory[lastIdx][0] * yHistory[lastIdx][0] +
        yHistory[lastIdx][1] * yHistory[lastIdx][1];
      if (yy > 0) H0 = sy / yy;
    }

    const r: [number, number] = [H0 * q[0], H0 * q[1]];
    for (let i = 0; i < sHistory.length; i++) {
      const bi =
        rhoHistory[i] * (yHistory[i][0] * r[0] + yHistory[i][1] * r[1]);
      r[0] += sHistory[i][0] * (alphas[i] - bi);
      r[1] += sHistory[i][1] * (alphas[i] - bi);
    }

    // Search direction (negative of L-BFGS direction)
    const d: [number, number] = [-r[0], -r[1]];

    // Backtracking line search with projection
    let alpha = 1.0;
    const c1 = 1e-4;
    const dirDeriv = gPrev[0] * d[0] + gPrev[1] * d[1];
    if (dirDeriv >= 0) {
      // Not a descent direction — use steepest descent
      d[0] = -gPrev[0];
      d[1] = -gPrev[1];
    }

    let xNew: [number, number] = [0, 0];
    let fNew = fPrev;
    let found = false;
    for (let ls = 0; ls < 20; ls++) {
      xNew = [
        clamp(x[0] + alpha * d[0], BOUNDS[0][0], BOUNDS[0][1]),
        clamp(x[1] + alpha * d[1], BOUNDS[1][0], BOUNDS[1][1]),
      ];
      fNew = evalObjective(xNew[0], xNew[1], psiSoil, parCost, parPhotosynth, parPlant, parEnv);
      const step = gPrev[0] * (xNew[0] - x[0]) + gPrev[1] * (xNew[1] - x[1]);
      if (fNew <= fPrev + c1 * step) {
        found = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!found) break;

    // Function value convergence
    if (Math.abs(fNew - fPrev) <= ftol * Math.max(1.0, Math.abs(fPrev))) break;

    const { g: gNew } = fdGradient(xNew, psiSoil, parCost, parPhotosynth, parPlant, parEnv);

    // Update L-BFGS history
    const s: [number, number] = [xNew[0] - x[0], xNew[1] - x[1]];
    const y: [number, number] = [gNew[0] - gPrev[0], gNew[1] - gPrev[1]];
    const sy = s[0] * y[0] + s[1] * y[1];
    if (sy > 1e-20) {
      sHistory.push(s);
      yHistory.push(y);
      rhoHistory.push(1.0 / sy);
      if (sHistory.length > m) {
        sHistory.shift();
        yHistory.shift();
        rhoHistory.shift();
      }
    }

    x = xNew;
    fPrev = fNew;
    gPrev[0] = gNew[0];
    gPrev[1] = gNew[1];
  }

  return { logJmax: x[0], dpsi: x[1] };
}

/**
 * Full P-Hydro solver: compute optimal photosynthesis-hydraulics state.
 *
 * Port of `pmodel_hydraulics_numerical` from phydro_mod.f90.
 * Uses projected L-BFGS with finite-difference gradients (matching Fortran).
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
  // Build parameter structs
  const parPlant: ParPlant = {
    conductivity: np.array(conductivityVal),
    psi50: np.array(psi50Val),
    b: np.array(bParam),
  };
  const parCost: ParCost = {
    alpha: np.array(alphaVal),
    gamma: np.array(gammaCost),
  };

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

  const parPhotosynth: ParPhotosynth = {
    kmm,
    gammastar: gs_star,
    phi0,
    Iabs,
    ca,
    patm: np.array(sp),
    delta: np.array(rdarkLeaf),
  };

  const viscWater = viscosityH2o(tcArr, spArr);
  const densWater = densityH2o(tcArr, spArr);

  const parEnv: ParEnv = {
    viscosityWater: viscWater,
    densityWater: densWater,
    patm: np.array(sp),
    tc: np.array(tc),
    vpd: np.array(vpd),
  };

  using psiSoilNp = np.array(psiSoilVal);

  // Optimise
  const opt = optimiseMidtermMulti(psiSoilNp, parCost, parPhotosynth, parPlant, parEnv);

  // Evaluate at optimum
  using logJmaxOpt = np.array(opt.logJmax);
  using dpsiOpt = np.array(opt.dpsi);
  const profit = fnProfit(
    logJmaxOpt,
    dpsiOpt,
    psiSoilNp,
    parCost,
    parPhotosynth,
    parPlant,
    parEnv,
    "PM",
    false,
  );

  const jmax = np.exp(logJmaxOpt);
  const dpsiOut = np.array(opt.dpsi);
  using gsVal = calcGs(dpsiOut, psiSoilNp, parPlant, parEnv);
  const gs = np.array(gsVal.item() as number); // clone to decouple lifetime
  gsVal; // disposed by using

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

  // Dispose parameter struct arrays
  parPlant.conductivity.dispose();
  parPlant.psi50.dispose();
  parPlant.b.dispose();
  parCost.alpha.dispose();
  parCost.gamma.dispose();
  parPhotosynth.kmm.dispose();
  parPhotosynth.gammastar.dispose();
  parPhotosynth.phi0.dispose();
  parPhotosynth.Iabs.dispose();
  parPhotosynth.ca.dispose();
  parPhotosynth.patm.dispose();
  parPhotosynth.delta.dispose();
  parEnv.viscosityWater.dispose();
  parEnv.densityWater.dispose();
  parEnv.patm.dispose();
  parEnv.tc.dispose();
  parEnv.vpd.dispose();

  return { jmax, dpsi: dpsiOut, gs, aj, ci, chi, vcmax, profit, chiJmaxLim };
}
