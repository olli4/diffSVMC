import { np } from "../precision.js";
import type { ParPhotosynth } from "./scale-conductivity.js";

/**
 * Electron-transport-limited CO2 assimilation rate.
 *
 * Fortran: `calc_assim_light_limited` in phydro_mod.f90
 * Solves: A = gs*(ca - ci) and A = φ₀*Iabs*jlim*(ci - Γ*)/(ci + 2Γ*)
 * via the quadratic formula.
 *
 * @param gs - Stomatal conductance (mol/m²/s)
 * @param jmax - Maximum electron transport rate (µmol/m²/s)
 * @param par - Photosynthesis parameters
 * @returns { ci, aj } — leaf internal CO2 (Pa) and assimilation rate (µmol/m²/s)
 */
export function calcAssimLightLimited(
  gs: np.Array,
  jmax: np.Array,
  par: ParPhotosynth,
): { ci: np.Array; aj: np.Array } {
  using ca = par.ca.ref;
  using _gs_tmp = gs.mul(1e6);
  using gs0 = _gs_tmp.div(par.patm); // convert to µmol/m²/s/Pa

  using phi0iabs = par.phi0.mul(par.Iabs);
  // jlim = phi0iabs / sqrt(1 + (4*phi0iabs/jmax)²)
  using fourPhi = phi0iabs.mul(4);
  using ratio = fourPhi.div(jmax);
  using ratio2 = ratio.mul(ratio);
  using _jaxTmp1 = ratio2.add(1);
  using denom = np.sqrt(_jaxTmp1);
  using jlim = phi0iabs.div(denom);
  using d = par.delta.ref;

  // Quadratic: A*ci² + B*ci + C = 0
  using A = gs0.mul(-1);
  using twoGamma = par.gammastar.mul(2);
  using _jaxTmp2 = gs0.mul(twoGamma);
  using _npConst1 = np.array(1);
  using _jaxTmp3 = _npConst1.sub(d);
  using _jaxTmp4 = jlim.mul(_jaxTmp3);
  using _B_1 = gs0.mul(ca);
  using _B_2 = _B_1.sub(_jaxTmp2);
  using B = _B_2.sub(_jaxTmp4);
  using _jaxTmp5 = d.mul(par.kmm);
  using _jaxTmp6 = par.gammastar.add(_jaxTmp5);
  using _jaxTmp7 = jlim.mul(_jaxTmp6);
  using _C_1 = gs0.mul(ca);
  using _C_2 = _C_1.mul(twoGamma);
  using C = _C_2.add(_jaxTmp7);

  // Solve: ci = q/A where q = -0.5*(B + sqrt(B² - 4AC))
  using _jaxTmp8_1 = A.mul(C);
  using _jaxTmp8 = _jaxTmp8_1.mul(4);
  using _disc_1 = B.mul(B);
  using disc = _disc_1.sub(_jaxTmp8);
  using _jaxTmp9 = np.array(0);
  using _jaxTmp10 = np.maximum(disc, _jaxTmp9);
  using sqrtDisc = np.sqrt(_jaxTmp10);
  using _q_1 = B.add(sqrtDisc);
  using q = _q_1.mul(-0.5);
  const ci = q.div(A);

  // aj = gs0 * (ca - ci)
  using _jaxTmp11 = ca.sub(ci);
  const aj = gs0.mul(_jaxTmp11);

  return { ci, aj };
}
