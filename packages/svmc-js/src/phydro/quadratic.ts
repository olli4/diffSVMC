import { np } from "../precision.js";

/**
 * Solve quadratic equation a·x² + b·x + c = 0 for root r1.
 *
 * Fortran: `quadratic` in phydro_mod.f90
 * Uses the Numerical Recipes formula: q = -0.5·(b + √(b²−4ac)), r1 = q/a.
 * Edge cases: a=0,b≠0 → -c/b; a=0,b=0 → 0.
 *
 * @param a - Quadratic coefficient
 * @param b - Linear coefficient
 * @param c - Constant term
 * @returns Root r1
 */
export function quadratic(
  a: np.Array,
  b: np.Array,
  c: np.Array,
): np.Array {
  using _fourAC = a.mul(c).mul(4);
  using _bb = b.mul(b);
  using _disc_raw = _bb.sub(_fourAC);
  // Single zero scalar reused for clamp, equality checks, and fallback result.
  // Inside jit() this is constant-folded; outside jit() it avoids 4 separate allocations.
  using zero = np.array(0);
  using disc = np.maximum(_disc_raw, zero);
  using sqrtDisc = np.sqrt(disc);
  using _bPlusSqrt = b.add(sqrtDisc);
  using q = _bPlusSqrt.mul(-0.5);

  // Standard quadratic root: q/a
  using r1_quad = q.div(a);

  // Linear fallback: -c/b (when a == 0)
  using _neg_c = c.mul(-1);
  using r1_linear = _neg_c.div(b);

  // Select: a==0 → linear (or 0 if b==0 too), else quadratic
  using _a_eq_zero = np.equal(a, zero);
  using _b_eq_zero = np.equal(b, zero);
  using _linear_or_zero = np.where(_b_eq_zero, zero, r1_linear);
  return np.where(_a_eq_zero, _linear_or_zero, r1_quad);
}
