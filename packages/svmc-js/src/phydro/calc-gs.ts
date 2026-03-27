import { np } from "../precision.js";
import { scaleConduct, type ParPlant, type ParEnv } from "./scale-conductivity.js";

/**
 * Stomatal conductance (mol/m²/s) given plant hydraulic traits.
 *
 * Fortran: `calc_gs` in phydro_mod.f90
 * Uses the approximate integral of the vulnerability curve.
 *
 * @param dpsi - Soil-to-leaf water potential difference (MPa)
 * @param psiSoil - Soil water potential (MPa)
 * @param parPlant - Plant hydraulic parameters
 * @param parEnv - Environmental parameters
 * @returns Stomatal conductance (mol/m²/s)
 */
export function calcGs(
  dpsi: np.Array,
  psiSoil: np.Array,
  parPlant: ParPlant,
  parEnv: ParEnv,
): np.Array {
  using K = scaleConduct(parPlant.conductivity, parEnv);
  using D = parEnv.vpd.div(parEnv.patm);

  // papprox = 0.5^((psi_soil - dpsi/2) / psi50)^b
  using halfDpsi = dpsi.div(2);
  using psiMid = psiSoil.sub(halfDpsi);
  using ratio = psiMid.div(parPlant.psi50);
  using exponent = np.power(ratio, parPlant.b);
  using _jaxTmp1 = np.array(0.5);
  using papprox = np.power(_jaxTmp1, exponent);

  // gs = K / 1.6 / D * dpsi * papprox
  using _k_scaled = K.div(1.6);
  using coeff = _k_scaled.div(D);
  using _gs_partial = coeff.mul(dpsi);
  return _gs_partial.mul(papprox);
}
