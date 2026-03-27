import { np } from "../precision.js";
import { ftempArrh } from "./ftemp-arrh.js";

/**
 * Michaelis-Menten coefficient for Rubisco-limited photosynthesis (Pa).
 *
 * Fortran: `calc_kmm` in phydro_mod.f90
 * References: Farquhar et al. (1980), Bernacchi et al. (2001)
 *
 * @param tc - Temperature (°C)
 * @param patm - Atmospheric pressure (Pa)
 * @returns Km (Pa)
 */
export function calcKmm(tc: np.Array, patm: np.Array): np.Array {
  const dhac = 79430; // Activation energy for Kc (J/mol)
  const dhao = 36380; // Activation energy for Ko (J/mol)
  const kco = 2.09476e5; // O2 partial pressure (ppm)
  const kc25 = 39.97; // Kc at 25°C (Pa)
  const ko25 = 27480; // Ko at 25°C (Pa)

  using tk = tc.add(273.15);
  using _npConst1 = np.array(kc25);
  using _dhac = np.array(dhac);
  using _kcArrh = ftempArrh(tk, _dhac);
  using kc = _npConst1.mul(_kcArrh);
  using _npConst2 = np.array(ko25);
  using _dhao = np.array(dhao);
  using _koArrh = ftempArrh(tk, _dhao);
  using ko = _npConst2.mul(_koArrh);

  // po = kco * 1e-6 * patm (O2 partial pressure)
  using po = patm.mul(kco * 1e-6);

  // kmm = kc * (1 + po/ko)
  using ratio = po.div(ko);
  using _jaxTmp1 = ratio.add(1);
  return kc.mul(_jaxTmp1);
}
