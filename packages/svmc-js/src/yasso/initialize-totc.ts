/**
 * Yasso20 C/N pool initialization from total carbon (initialize_totc).
 *
 * Fortran reference: vendor/SVMC/src/yasso.f90 L184-236
 *
 * Initialises the 5-pool AWENH carbon state and the single nitrogen pool
 * given a prescribed total carbon. The output is a weighted blend of:
 *   - an equilibrium partitioning (from a steady-state solve of A*x = -input)
 *   - a "legacy" partitioning (all carbon in the H pool)
 *
 * Dependencies:
 *   - evaluateMatrixMeanTempr: builds the 5x5 AWENH matrix using 4-point
 *     annual-cycle temperature averaging.
 *   - evalSteadystateNitr: iterative CUE-based steady-state N solver.
 *   - np.linalg.solve: LU-based linear system solver.
 *
 * All operations stay on traced np.Array values.
 * Forward-only: jax-js-nonconsuming does not support reverse-mode
 * differentiation through np.minimum/np.maximum inside lax.foriLoop.
 */

import { lax } from "@hamk-uas/jax-js-nonconsuming";
import { np } from "../precision.js";

const DAYS_YR = 365.25;
const NC_MB = 0.1;      // microbial biomass N:C ratio
const CUE_MIN = 0.1;    // minimum carbon use efficiency
const NC_H_MAX = 0.1;   // maximum N:C ratio of humus pool
const PI = 3.141592653589793;
const MAX_CUE_ITER = 10;

// AWENH fractions — currently identical for fineroot & leaf in upstream Fortran.
const AWENH_FINEROOT = [0.46, 0.32, 0.04, 0.18, 0.0];
const AWENH_LEAF = [0.46, 0.32, 0.04, 0.18, 0.0];

/** Shorthand: extract traced scalar element i from a 1-D array. */
function el(arr: np.Array, i: number): np.Array {
  return arr.slice(i);
}

function requireUnitIntervalChecked(name: string, value: np.Array): void {
  const scalar = value.item() as number;
  if (scalar < 0.0 || scalar > 1.0) {
    throw new RangeError(`${name} must be in [0, 1], got ${scalar}`);
  }
}

/**
 * Build the 5x5 AWENH decomposition matrix using 4-point temperature averaging.
 *
 * Fortran reference: vendor/SVMC/src/yasso.f90 L401-464
 *
 * Unlike the daily-step version (evaluateMatrix in decompose.ts), this uses
 * 4 strategically placed temperature points to approximate the annual cycle
 * instead of a single daily temperature.
 */
function evaluateMatrixMeanTempr(
  param: np.Array,
  tempr: np.Array,
  precipYr: np.Array,
  temprAmpl: np.Array,
): np.Array {
  // 4-point annual cycle approximation
  const sqrt2 = Math.sqrt(2.0);
  const c1 = 4 * (1 / sqrt2 - 1) / PI;
  const c2 = -4 / sqrt2 / PI;
  const c3 = 4 * (1 - 1 / sqrt2) / PI;
  const c4 = 4 / sqrt2 / PI;

  using _te0a = temprAmpl.mul(c1); using te0 = _te0a.add(tempr);
  using _te1a = temprAmpl.mul(c2); using te1 = _te1a.add(tempr);
  using _te2a = temprAmpl.mul(c3); using te2 = _te2a.add(tempr);
  using _te3a = temprAmpl.mul(c4); using te3 = _te3a.add(tempr);
  using te = np.stack([te0, te1, te2, te3]);

  // Temperature modifiers: average exp(beta1*T + beta2*T^2) over 4 points
  using te2sq = np.power(te, 2);
  using p21 = el(param, 21); using p22 = el(param, 22);
  using p23 = el(param, 23); using p24 = el(param, 24);
  using p25 = el(param, 25); using p26 = el(param, 26);

  using _ea1 = p21.mul(te); using _ea2 = p22.mul(te2sq); using _ea = _ea1.add(_ea2);
  using _en1 = p23.mul(te); using _en2 = p24.mul(te2sq); using _en = _en1.add(_en2);
  using _eh1 = p25.mul(te); using _eh2 = p26.mul(te2sq); using _eh = _eh1.add(_eh2);
  using _expa = np.exp(_ea); using _suma = np.sum(_expa); using temprm = _suma.mul(0.25);
  using _expn = np.exp(_en); using _sumn = np.sum(_expn); using temprmN = _sumn.mul(0.25);
  using _exph = np.exp(_eh); using _sumh = np.sum(_exph); using temprmH = _sumh.mul(0.25);

  // Precipitation modifiers (no /12 division)
  using precipK = precipYr.mul(0.001);
  using p27 = el(param, 27);
  using _pf1 = p27.mul(precipK); using _pf2 = np.exp(_pf1); using _pf3 = _pf2.neg();
  using precFac = _pf3.add(1);
  using p28 = el(param, 28);
  using _pfn1 = p28.mul(precipK); using _pfn2 = np.exp(_pfn1); using _pfn3 = _pfn2.neg();
  using precFacN = _pfn3.add(1);
  using p29 = el(param, 29);
  using _pfh1 = p29.mul(precipK); using _pfh2 = np.exp(_pfh1); using _pfh3 = _pfh2.neg();
  using precFacH = _pfh3.add(1);

  using decm = temprm.mul(precFac);
  using decmN = temprmN.mul(precFacN);
  using decmH = temprmH.mul(precFacH);

  // Diagonal decomposition rates
  using e0 = el(param, 0);  using alpha0 = np.abs(e0);
  using e1 = el(param, 1);  using alpha1 = np.abs(e1);
  using e2 = el(param, 2);  using alpha2 = np.abs(e2);
  using e3 = el(param, 3);  using alpha3 = np.abs(e3);
  using e31 = el(param, 31); using alphaH = np.abs(e31);

  using _dr0 = alpha0.mul(decm);  using dr0 = _dr0.neg();
  using _dr1 = alpha1.mul(decm);  using dr1 = _dr1.neg();
  using _dr2 = alpha2.mul(decm);  using dr2 = _dr2.neg();
  using _dr3 = alpha3.mul(decmN); using dr3 = _dr3.neg();
  using _dr4 = alphaH.mul(decmH); using dr4 = _dr4.neg();

  using ad0 = np.abs(dr0);
  using ad1 = np.abs(dr1);
  using ad2 = np.abs(dr2);
  using ad3 = np.abs(dr3);

  // Transfer-fraction parameters
  using p4 = el(param, 4);   using p5 = el(param, 5);   using p6 = el(param, 6);
  using p7 = el(param, 7);   using p8 = el(param, 8);   using p9 = el(param, 9);
  using p10 = el(param, 10); using p11 = el(param, 11); using p12 = el(param, 12);
  using p13 = el(param, 13); using p14 = el(param, 14); using p15 = el(param, 15);
  using p30 = el(param, 30);

  using zero = np.array(0);

  // Assemble rows (no leaching terms, no size dependence)
  using _r01 = p4.mul(ad1);  using _r02 = p5.mul(ad2);  using _r03 = p6.mul(ad3);
  using r0 = np.stack([dr0, _r01, _r02, _r03, zero]);
  using _r10 = p7.mul(ad0);  using _r12 = p8.mul(ad2);  using _r13 = p9.mul(ad3);
  using r1 = np.stack([_r10, dr1, _r12, _r13, zero]);
  using _r20 = p10.mul(ad0); using _r21 = p11.mul(ad1); using _r23 = p12.mul(ad3);
  using r2 = np.stack([_r20, _r21, dr2, _r23, zero]);
  using _r30 = p13.mul(ad0); using _r31 = p14.mul(ad1); using _r32 = p15.mul(ad2);
  using r3 = np.stack([_r30, _r31, _r32, dr3, zero]);
  using _r40 = p30.mul(ad0); using _r41 = p30.mul(ad1);
  using _r42 = p30.mul(ad2); using _r43 = p30.mul(ad3);
  using r4 = np.stack([_r40, _r41, _r42, _r43, dr4]);

  return np.stack([r0, r1, r2, r3, r4]);
}

/**
 * Evaluate steady-state nitrogen pool via CUE iteration.
 *
 * Fortran reference: vendor/SVMC/src/yasso.f90 L151-182
 *
 * Iterates MAX_CUE_ITER times to converge the CUE-based nitrogen balance.
 * Uses lax.foriLoop for structural parity with JAX.
 * Forward-only: np.minimum/np.maximum inside the loop body are not
 * reverse-mode differentiable in jax-js-nonconsuming.
 */
function evalSteadystateNitr(
  cstate: np.Array,
  respYr: np.Array,
  nitrInputYr: np.Array,
  matrix: np.Array,
): np.Array {
  using m44 = matrix.slice(4, 4);
  using cs4 = cstate.slice(4);
  using decompH = m44.mul(cs4);

  using sumAll = np.sum(cstate);
  using sumAwen = sumAll.sub(cs4);

  const cueInit = np.array(0.43);

  const cueFinal = lax.foriLoop(0, MAX_CUE_ITER, (_, cue) => {
    using oneMinusCue = np.array(1.0).sub(cue);
    using _rm = respYr.sub(decompH);
    using cuptAwen = _rm.div(oneMinusCue);

    // nc_awen = (1/cupt) * (nc_mb*cue*cupt - nc_h_max*decomp_h + nitr_input_yr)
    using _t1a = cue.mul(cuptAwen); using _term1 = _t1a.mul(NC_MB);
    using _term2 = decompH.mul(NC_H_MAX);
    using _sub12 = _term1.sub(_term2); using _inner = _sub12.add(nitrInputYr);
    using invCupt = np.array(1.0).div(cuptAwen);
    using ncAwen = invCupt.mul(_inner);

    using _nAwen = sumAwen.mul(ncAwen);
    using _nH = cs4.mul(NC_H_MAX);
    using nstate = _nAwen.add(_nH);
    using ncSom = nstate.div(sumAll);

    // CUE = clip(0.43 * (nc_som / nc_mb)^0.6, cue_min, 1.0)
    using _ratio = ncSom.div(NC_MB);
    using _power = np.power(_ratio, 0.6);
    using _raw = _power.mul(0.43);
    using _one = np.array(1.0); using _capped = np.minimum(_raw, _one);
    using _cueMin = np.array(CUE_MIN);
    return np.maximum(_capped, _cueMin);
  }, cueInit);

  // Recompute nstate from converged CUE
  using oneMinusCueFinal = np.array(1.0).sub(cueFinal);
  using _rm2 = respYr.sub(decompH);
  using cuptFinal = _rm2.div(oneMinusCueFinal);
  using _t1f1 = cueFinal.mul(cuptFinal); using _t1f = _t1f1.mul(NC_MB);
  using _t2f = decompH.mul(NC_H_MAX);
  using _sub12f = _t1f.sub(_t2f); using _innerF = _sub12f.add(nitrInputYr);
  using invCuptF = np.array(1.0).div(cuptFinal);
  using ncAwenF = invCuptF.mul(_innerF);
  using _nHf = cs4.mul(NC_H_MAX);
  using _nAwenF = sumAwen.mul(ncAwenF);
  const nstate = _nAwenF.add(_nHf);

  cueFinal.dispose();
  return nstate;
}

/**
 * Initialise Yasso C/N pools from total carbon.
 *
 * Fortran reference: vendor/SVMC/src/yasso.f90 L184-236
 *
 * The output is a weighted blend of:
 *   - equilibrium: solve A*x = -input for unit input, scale to match totc
 *   - legacy: all carbon in H pool, nitrogen at nc_h_max
 *
 * Note: In the current Fortran parameterisation, awenh_fineroot == awenh_leaf,
 * so fract_root_input has no effect.
 *
 * Pure traced compute kernel. Callers that need eager host-side validation
 * should use initializeTotcChecked instead of calling this function directly.
 */
export function initializeTotc(
  param: np.Array,
  totc: np.Array,
  cnInput: np.Array,
  fractRootInput: np.Array,
  fractLegacySoc: np.Array,
  temprC: np.Array,
  precipDay: np.Array,
  temprAmpl: np.Array,
): { cstate: np.Array; nstate: np.Array } {
  // Build matrix from mean temperature
  using precipYr = precipDay.mul(DAYS_YR);
  using matrix = evaluateMatrixMeanTempr(param, temprC, precipYr, temprAmpl);

  // Blend unit input composition
  using _negRoot = fractRootInput.neg();
  using _oneMinusRoot = _negRoot.add(1.0);
  using _root0 = fractRootInput.mul(AWENH_FINEROOT[0]);
  using _root1 = fractRootInput.mul(AWENH_FINEROOT[1]);
  using _root2 = fractRootInput.mul(AWENH_FINEROOT[2]);
  using _root3 = fractRootInput.mul(AWENH_FINEROOT[3]);
  using _root4 = fractRootInput.mul(AWENH_FINEROOT[4]);
  using _leaf0 = _oneMinusRoot.mul(AWENH_LEAF[0]);
  using _leaf1 = _oneMinusRoot.mul(AWENH_LEAF[1]);
  using _leaf2 = _oneMinusRoot.mul(AWENH_LEAF[2]);
  using _leaf3 = _oneMinusRoot.mul(AWENH_LEAF[3]);
  using _leaf4 = _oneMinusRoot.mul(AWENH_LEAF[4]);
  using _unit0 = _root0.add(_leaf0);
  using _unit1 = _root1.add(_leaf1);
  using _unit2 = _root2.add(_leaf2);
  using _unit3 = _root3.add(_leaf3);
  using _unit4 = _root4.add(_leaf4);
  using unitInput = np.stack([_unit0, _unit1, _unit2, _unit3, _unit4]);

  // Solve for equilibrium partitioning: A * tmpstate = -unit_input
  using inputZero = unitInput.mul(0.0);
  using negInput = inputZero.sub(unitInput);
  using tmpstate = np.linalg.solve(matrix, negInput);
  using sumTmp = np.sum(tmpstate);
  using eqfac = totc.div(sumTmp);
  using eqstate = tmpstate.mul(eqfac);

  // Equilibrium nitrogen via CUE iteration
  // resp_yr = eqfac (steady state: resp == input)
  using nitrInputYr = eqfac.div(cnInput);
  using eqnitr = evalSteadystateNitr(eqstate, eqfac, nitrInputYr, matrix);

  // Blend equilibrium and legacy
  using _legacyZero = totc.mul(0.0);
  using _legacyOne = _legacyZero.add(1.0);
  using _legacyState = np.stack([
    _legacyZero,
    _legacyZero,
    _legacyZero,
    _legacyZero,
    _legacyOne,
  ]);
  using _legacyC1 = _legacyState.mul(totc);
  using _legacyC = _legacyC1.mul(fractLegacySoc);
  using _oneMinusLegacy = np.array(1.0).sub(fractLegacySoc);
  using _eqC = eqstate.mul(_oneMinusLegacy);
  const cstate = _legacyC.add(_eqC);

  using _legacyN1 = totc.mul(NC_H_MAX);
  using _legacyN = _legacyN1.mul(fractLegacySoc);
  using _eqN = eqnitr.mul(_oneMinusLegacy);
  const nstate = _legacyN.add(_eqN);

  return { cstate, nstate };
}

/**
 * Validated eager wrapper for initializeTotc.
 *
 * This mirrors the Fortran input-range guards for host scalar callers and is
 * intentionally not tracing-friendly because it extracts JS scalars via item().
 */
export function initializeTotcChecked(
  param: np.Array,
  totc: np.Array,
  cnInput: np.Array,
  fractRootInput: np.Array,
  fractLegacySoc: np.Array,
  temprC: np.Array,
  precipDay: np.Array,
  temprAmpl: np.Array,
): { cstate: np.Array; nstate: np.Array } {
  requireUnitIntervalChecked("fractRootInput", fractRootInput);
  requireUnitIntervalChecked("fractLegacySoc", fractLegacySoc);

  return initializeTotc(
    param,
    totc,
    cnInput,
    fractRootInput,
    fractLegacySoc,
    temprC,
    precipDay,
    temprAmpl,
  );
}
