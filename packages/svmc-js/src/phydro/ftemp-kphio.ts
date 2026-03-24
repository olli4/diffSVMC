import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

/**
 * Temperature dependence of quantum yield efficiency.
 *
 * Fortran: `ftemp_kphio` in phydro_mod.f90
 * References: Bernacchi et al. (2003)
 *
 * @param tc - Temperature (°C)
 * @param c4 - Whether C4 photosynthesis (default false → C3)
 * @returns Scaling factor for quantum yield (dimensionless, clamped ≥ 0)
 */
export function ftempKphio(tc: np.Array, c4 = false): np.Array {
  if (c4) {
    // C4: -0.064 + 0.03*tc - 0.000464*tc²
    using t2 = tc.mul(tc);
    using term1 = tc.mul(0.03);
    using term2 = t2.mul(0.000464);
    using raw = term1.sub(term2).sub(0.064);
    using _jaxTmp1 = np.array(0);
    return np.maximum(raw, _jaxTmp1);
  } else {
    // C3: 0.352 + 0.022*tc - 3.4e-4*tc²
    using t2 = tc.mul(tc);
    using term1 = tc.mul(0.022);
    using term2 = t2.mul(3.4e-4);
    using raw = term1.sub(term2).add(0.352);
    using _jaxTmp2 = np.array(0);
    return np.maximum(raw, _jaxTmp2);
  }
}
