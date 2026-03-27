/**
 * Yasso20 yearly ODE solver (mod5c20).
 *
 * Fortran reference: vendor/SVMC/src/yassofortran20.f90 L24-161
 *
 * Solves the linear ODE: x'(t) = A(theta) * x(t) + b, x(0) = init
 *   - Transient: x(t) = A^-1 * (exp(At) * (A*init + b) - b)
 *   - Steady state: x_inf = -A^-1 * b
 *
 * Coefficient matrix A is built from Yasso20 parameters theta, monthly
 * temperatures, annual precipitation, size class, and leaching rate.
 *
 * All operations stay on traced np.Array values so the full computation
 * is differentiable via jax-js-nonconsuming autodiff (jit/grad/vjp).
 */

import { getNumericDType, np } from "../precision.js";
import { matrixExp } from "./matrixexp.js";

// Fortran uses 1e-12 (float64).  In float32 the transient solver suffers
// catastrophic cancellation when ||At|| is very small (e.g. -80 °C →
// tem ≈ 3e-8), so we widen the threshold to 1e-6 for that dtype.
const TOL = getNumericDType() === "float32" ? 1e-6 : 1e-12;

/** Shorthand: extract traced scalar element i from a 1-D array. */
function el(arr: np.Array, i: number): np.Array {
  return arr.slice(i);
}

/**
 * Build the 5×5 AWENH coefficient matrix A and temperature multiplier tem.
 *
 * Everything stays on traced np.Array for autodiff. Uses vectorized
 * np.exp / np.sum over the 12-element temp array, then assembles the
 * matrix row-by-row with np.stack.
 *
 * Fortran: mod5c20 matrix-construction block, L70–145
 */
function buildCoefficientMatrix(
  theta: np.Array,
  temp: np.Array,
  prec: np.Array,
  d: np.Array,
  leac: np.Array,
): { A: np.Array; tem: np.Array } {
  // Temperature dependence: sum(exp(a*T + b*T²)) over 12 months
  using t2 = np.power(temp, 2);

  using e21 = el(theta, 21);  using _ta1 = e21.mul(temp);
  using e22 = el(theta, 22);  using _ta2 = e22.mul(t2);
  using temArg = _ta1.add(_ta2);
  using e23 = el(theta, 23);  using _tn1 = e23.mul(temp);
  using e24 = el(theta, 24);  using _tn2 = e24.mul(t2);
  using temNArg = _tn1.add(_tn2);
  using e25 = el(theta, 25);  using _th1 = e25.mul(temp);
  using e26 = el(theta, 26);  using _th2 = e26.mul(t2);
  using temHArg = _th1.add(_th2);

  using _temExp = np.exp(temArg);    using temSum = np.sum(_temExp);
  using _temNExp = np.exp(temNArg);  using temNSum = np.sum(_temNExp);
  using _temHExp = np.exp(temHArg);  using temHSum = np.sum(_temHExp);

  // Precipitation dependence: tem * (1 - exp(theta * prec/1000)) / 12
  using precK = prec.div(1000);
  using e27 = el(theta, 27);
  using _pf1 = e27.mul(precK);  using _pf2 = np.exp(_pf1);  using _pf3 = _pf2.neg();
  using precFac = _pf3.add(1);
  using e28 = el(theta, 28);
  using _pfn1 = e28.mul(precK); using _pfn2 = np.exp(_pfn1); using _pfn3 = _pfn2.neg();
  using precFacN = _pfn3.add(1);
  using e29 = el(theta, 29);
  using _pfh1 = e29.mul(precK); using _pfh2 = np.exp(_pfh1); using _pfh3 = _pfh2.neg();
  using precFacH = _pfh3.add(1);

  using _tem1 = temSum.mul(precFac);   const tem = _tem1.div(12);
  using _temN1 = temNSum.mul(precFacN); using temN = _temN1.div(12);
  using _temH1 = temHSum.mul(precFacH); using temH = _temH1.div(12);

  // Size class dependence: min(1, (1 + th32*d + th33*d²)^-|th34|)
  using d2 = np.power(d, 2);
  using e32 = el(theta, 32);  using _sb1 = e32.mul(d);
  using e33 = el(theta, 33);  using _sb2 = e33.mul(d2);
  using _sb3 = _sb1.add(_sb2);        using sizeBase = _sb3.add(1);
  using e34 = el(theta, 34);  using _absE34 = np.abs(e34);
  using sizeExp = _absE34.neg();
  using sizePow = np.power(sizeBase, sizeExp);
  using _one = np.array(1);
  using sizeDep = np.minimum(_one, sizePow);

  // Diagonal decomposition rates (AWEN + H)
  using e0 = el(theta, 0);   using alpha0 = np.abs(e0);
  using e1 = el(theta, 1);   using alpha1 = np.abs(e1);
  using e2 = el(theta, 2);   using alpha2 = np.abs(e2);
  using e3 = el(theta, 3);   using alpha3 = np.abs(e3);
  using e31 = el(theta, 31); using alphaH = np.abs(e31);

  using _dr0a = alpha0.mul(tem);  using _dr0b = _dr0a.mul(sizeDep);
  using dr0 = _dr0b.neg();
  using _dr1a = alpha1.mul(tem);  using _dr1b = _dr1a.mul(sizeDep);
  using dr1 = _dr1b.neg();
  using _dr2a = alpha2.mul(tem);  using _dr2b = _dr2a.mul(sizeDep);
  using dr2 = _dr2b.neg();
  using _dr3a = alpha3.mul(temN); using _dr3b = _dr3a.mul(sizeDep);
  using dr3 = _dr3b.neg();
  using _dr4a = alphaH.mul(temH);
  using dr4 = _dr4a.neg();

  using ad0 = np.abs(dr0);
  using ad1 = np.abs(dr1);
  using ad2 = np.abs(dr2);
  using ad3 = np.abs(dr3);

  // Leaching (AWEN only)
  using leacTerm = leac.mul(precK);

  // Extract transfer-fraction parameters
  using p4 = el(theta, 4);   using p5 = el(theta, 5);   using p6 = el(theta, 6);
  using p7 = el(theta, 7);   using p8 = el(theta, 8);   using p9 = el(theta, 9);
  using p10 = el(theta, 10); using p11 = el(theta, 11); using p12 = el(theta, 12);
  using p13 = el(theta, 13); using p14 = el(theta, 14); using p15 = el(theta, 15);
  using p30 = el(theta, 30);

  using zero = np.array(0);

  // Assemble rows then stack into 5×5
  using _r00 = dr0.add(leacTerm);  using _r01 = p4.mul(ad1);
  using _r02 = p5.mul(ad2);   using _r03 = p6.mul(ad3);
  using r0 = np.stack([_r00, _r01, _r02, _r03, zero]);
  using _r10 = p7.mul(ad0);   using _r11 = dr1.add(leacTerm);
  using _r12 = p8.mul(ad2);   using _r13 = p9.mul(ad3);
  using r1 = np.stack([_r10, _r11, _r12, _r13, zero]);
  using _r20 = p10.mul(ad0);  using _r21 = p11.mul(ad1);
  using _r22 = dr2.add(leacTerm); using _r23 = p12.mul(ad3);
  using r2 = np.stack([_r20, _r21, _r22, _r23, zero]);
  using _r30 = p13.mul(ad0);  using _r31 = p14.mul(ad1);
  using _r32 = p15.mul(ad2);  using _r33 = dr3.add(leacTerm);
  using r3 = np.stack([_r30, _r31, _r32, _r33, zero]);
  using _r40 = p30.mul(ad0);  using _r41 = p30.mul(ad1);
  using _r42 = p30.mul(ad2);  using _r43 = p30.mul(ad3);
  using r4 = np.stack([_r40, _r41, _r42, _r43, dr4]);

  const A = np.stack([r0, r1, r2, r3, r4]);

  return { A, tem };
}

/**
 * Yasso20 yearly ODE solver.
 *
 * Fortran: mod5c20 in yassofortran20.f90 L24-161
 *
 * @param theta - Parameter vector (35,).
 * @param time - Integration time (years), scalar.
 * @param temp - Monthly mean temperatures (12,).
 * @param prec - Annual precipitation (mm), scalar.
 * @param init - Initial AWENH state (5,).
 * @param b - Annual C input (5,).
 * @param d - Size class, scalar.
 * @param leac - Leaching parameter, scalar.
 * @param steadystatePredict - If true, compute steady-state x = -A^-1*b.
 * @returns AWENH state (5,) after integration. Caller must dispose.
 */
export function mod5c20Fn(
  theta: np.Array,
  time: np.Array,
  temp: np.Array,
  prec: np.Array,
  init: np.Array,
  b: np.Array,
  d: np.Array,
  leac: np.Array,
  steadystatePredict: boolean = false,
): np.Array {
  const { A, tem } = buildCoefficientMatrix(theta, temp, prec, d, leac);

  // tem is only needed for the conditional check
  using temCond = tem.lessEqual(TOL);
  tem.dispose();

  // Both branches computed; np.where selects based on tem and steadystate.
  // This keeps the computation graph intact for autodiff.
  // Early return path: very cold / no rain → no decomposition → init + b*time
  using _bt = b.mul(time);
  using earlyResult = init.add(_bt);

  // Steady state: x = -A^-1 * b → solve(-A, b)
  using negA = np.negative(A);
  using steadyResult = np.linalg.solve(negA, b);

  // Transient: x(t) = A^-1 * (exp(At) * (A*init + b) - b)
  using _ainit = np.matmul(A, init);
  using z1 = _ainit.add(b);
  using At = A.mul(time);
  using mexpAt = matrixExp(At);
  using _mz1 = np.matmul(mexpAt, z1);
  using z2 = _mz1.sub(b);
  using transientResult = np.linalg.solve(A, z2);

  A.dispose();

  // Select: steadystate_pred ? steady : transient
  using normalResult = np.where(steadystatePredict, steadyResult, transientResult);

  // Select: tem <= TOL ? early : normal
  return np.where(temCond, earlyResult, normalResult);
}
