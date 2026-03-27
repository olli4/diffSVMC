import { np } from "../precision.js";

/**
 * Arrhenius-type temperature response function.
 *
 * Fortran: `ftemp_arrh` in phydro_mod.f90
 *
 * @param tk - Air temperature (K)
 * @param dha - Activation energy (J/mol)
 * @returns Temperature scaling factor (dimensionless)
 */
export function ftempArrh(tk: np.Array, dha: np.Array): np.Array {
  const kR = 8.3145; // Universal gas constant, J/mol/K
  const tkref = 298.15; // Reference temperature, K

  // ftemp_arrh = exp(dha * (tk - tkref) / (tkref * kR * tk))
  using num = tk.sub(tkref).mul(dha);
  using _npConst1 = np.array(tkref * kR);
  using den = _npConst1.mul(tk);
  using ratio = num.div(den);
  return np.exp(ratio);
}
