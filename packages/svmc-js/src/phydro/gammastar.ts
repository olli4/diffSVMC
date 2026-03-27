import { np } from "../precision.js";
import { ftempArrh } from "./ftemp-arrh.js";

/**
 * Photorespiratory CO2 compensation point Γ* (Pa).
 *
 * Fortran: `gammastar` in phydro_mod.f90
 * References: Bernacchi et al. (2001)
 *
 * @param tc - Temperature (°C)
 * @param patm - Atmospheric pressure (Pa)
 * @returns Γ* in Pa
 */
export function gammastar(tc: np.Array, patm: np.Array): np.Array {
  const dha = 37830; // Activation energy (J/mol)
  const gs25_0 = 4.332; // Γ* at 25°C, 101325 Pa (Pa)
  const patm0 = 101325; // Standard sea-level pressure (Pa)

  using tk = tc.add(273.15);
  using _dha = np.array(dha);
  using arrh = ftempArrh(tk, _dha);
  using _npConst1 = np.array(gs25_0);
  using _scaled_1 = _npConst1.mul(patm);
  using scaled = _scaled_1.div(patm0);
  return scaled.mul(arrh);
}
