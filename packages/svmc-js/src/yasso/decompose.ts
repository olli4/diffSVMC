/**
 * Yasso20 daily C/N decomposition step (decompose).
 *
 * Fortran reference: vendor/SVMC/src/yasso.f90 L255-318
 *
 * Computes daily carbon and nitrogen tendencies using first-order explicit
 * Euler integration of the 5x5 AWENH decomposition matrix. Unlike mod5c20,
 * this uses a single daily temperature (not monthly averages), has no size
 * class dependence, and no leaching terms.
 *
 * Nitrogen dynamics follow CUE-based immobilization/mineralization.
 *
 * All operations stay on traced np.Array values so the full computation is
 * differentiable via jax-js-nonconsuming autodiff (jit/grad/vjp).
 */

import { np } from "../precision.js";

const DAYS_YR = 365.25;
const NC_MB = 0.1;      // microbial biomass N:C ratio
const CUE_MIN = 0.1;    // minimum carbon use efficiency
const NC_H_MAX = 0.1;   // maximum N:C ratio of humus pool

/** Shorthand: extract traced scalar element i from a 1-D array. */
function el(arr: np.Array, i: number): np.Array {
  return arr.slice(i);
}

/**
 * Build the 5x5 AWENH decomposition matrix A for daily decomposition.
 *
 * Fortran reference: vendor/SVMC/src/yasso.f90 L339-391
 *
 * Unlike the mod5c20 matrix construction, this version:
 *   - Uses a single daily temperature (no monthly averaging, no /12)
 *   - Has no size class dependence
 *   - Has no leaching terms
 */
function evaluateMatrix(param: np.Array, tempr: np.Array, precipYr: np.Array): np.Array {
  // Temperature modifiers: exp(beta1*T + beta2*T^2)
  using t2 = np.power(tempr, 2);

  using e21 = el(param, 21); using _ta1 = e21.mul(tempr);
  using e22 = el(param, 22); using _ta2 = e22.mul(t2);
  using temprArg = _ta1.add(_ta2);
  using e23 = el(param, 23); using _tn1 = e23.mul(tempr);
  using e24 = el(param, 24); using _tn2 = e24.mul(t2);
  using temprnArg = _tn1.add(_tn2);
  using e25 = el(param, 25); using _th1 = e25.mul(tempr);
  using e26 = el(param, 26); using _th2 = e26.mul(t2);
  using temprhArg = _th1.add(_th2);

  using temprm = np.exp(temprArg);
  using temprmN = np.exp(temprnArg);
  using temprmH = np.exp(temprhArg);

  // Precipitation modifiers: temprm * (1 - exp(gamma * precip * 0.001))
  // No /12 division (unlike mod5c20)
  using precipK = precipYr.mul(0.001);
  using e27 = el(param, 27);
  using _pf1 = e27.mul(precipK); using _pf2 = np.exp(_pf1); using _pf3 = _pf2.neg();
  using precFac = _pf3.add(1);
  using e28 = el(param, 28);
  using _pfn1 = e28.mul(precipK); using _pfn2 = np.exp(_pfn1); using _pfn3 = _pfn2.neg();
  using precFacN = _pfn3.add(1);
  using e29 = el(param, 29);
  using _pfh1 = e29.mul(precipK); using _pfh2 = np.exp(_pfh1); using _pfh3 = _pfh2.neg();
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
 * Daily C/N decomposition step.
 *
 * Fortran reference: vendor/SVMC/src/yasso.f90 L255-318
 *
 * Computes carbon tendencies via first-order Euler integration:
 *   ctend = A * cstate * (timestep_days / 365.25)
 * and nitrogen tendency via CUE-based immobilization/mineralization.
 *
 * All branches use np.where for differentiability.
 *
 * @param param - Parameter vector (35,).
 * @param timestepDays - Timestep length (days), scalar.
 * @param temprC - Air temperature (deg C), scalar.
 * @param precipDay - Precipitation (mm/day), scalar.
 * @param cstate - AWENH carbon state (5,).
 * @param nstate - Nitrogen state (single pool), scalar.
 * @returns { ctend, ntend } - Carbon tendencies (5,) and nitrogen tendency (scalar).
 *   Caller must dispose both.
 */
export function decomposeFn(
  param: np.Array,
  timestepDays: np.Array,
  temprC: np.Array,
  precipDay: np.Array,
  cstate: np.Array,
  nstate: np.Array,
): { ctend: np.Array; ntend: np.Array } {
  // Carbon: explicit Euler
  using precipYr = precipDay.mul(DAYS_YR);
  using matrix = evaluateMatrix(param, temprC, precipYr);
  using timestepYr = timestepDays.div(DAYS_YR);
  using _ac = np.matmul(matrix, cstate);
  const ctend = _ac.mul(timestepYr);
  using negCtend = np.negative(ctend);
  using resp = np.sum(negCtend);

  // Total carbon
  using totc = np.sum(cstate);

  // Nitrogen dynamics (branch-free for autodiff)
  // PORT-BRANCH: totc < 1e-6 -> ntend = 0
  using m44 = matrix.slice(4, 4);  // matrix[4,4]
  using cs4 = cstate.slice(4);     // cstate[4]
  using _dh1 = m44.mul(cs4);
  using decompH = _dh1.mul(timestepYr);

  // PORT-BRANCH: cstate[4]*nc_h_max > nstate -> nc_h = nstate/totc
  using _guard = totc.add(1e-30);
  using ncHUnusual = nstate.div(_guard);
  using ncHNormal = np.array(NC_H_MAX);
  using _checkH = cs4.mul(NC_H_MAX);
  using _condH = _checkH.greater(nstate);
  using ncH = np.where(_condH, ncHUnusual, ncHNormal);

  using _nitr1 = cs4.mul(ncH);
  using nitrAwen = nstate.sub(_nitr1);
  using _denom = totc.sub(cs4);
  using _denom2 = _denom.add(1e-9);
  using ncAwen = nitrAwen.div(_denom2);
  using ncSom = nstate.div(_guard);

  // PORT-BRANCH: CUE clamped to [cue_min, 1.0]
  using _ratio = ncSom.div(NC_MB);
  using _power = np.power(_ratio, 0.6);
  using _cueRaw = _power.mul(0.43);
  using _cueMax = np.minimum(_cueRaw, np.array(1.0));
  using cue = np.maximum(_cueMax, np.array(CUE_MIN));

  using _oneMinusCue = np.array(1.0).sub(cue);
  using _respMinusDH = resp.sub(decompH);
  using cuptAwen = _respMinusDH.div(_oneMinusCue);
  using growthC = cue.mul(cuptAwen);

  // ntend = nc_mb * growth_c - nc_awen * cupt_awen - nc_h * decomp_h
  using _t1 = np.array(NC_MB).mul(growthC);
  using _t2 = ncAwen.mul(cuptAwen);
  using _t3 = ncH.mul(decompH);
  using _t12 = _t1.sub(_t2);
  using ntendComputed = _t12.sub(_t3);

  // PORT-BRANCH: totc < 1e-6 -> ntend = 0
  using _zeroN = np.array(0);
  using _condLow = totc.less(1e-6);
  const ntend = np.where(_condLow, _zeroN, ntendComputed);

  return { ctend, ntend };
}
