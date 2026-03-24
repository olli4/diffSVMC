import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

/**
 * Density of water (kg/m³) as a function of temperature and pressure.
 * Uses the Tumlirz Equation.
 *
 * Fortran: `density_h2o` in phydro_mod.f90
 * References: Fisher & Dial (1975)
 *
 * @param tc - Temperature (°C)
 * @param patm - Atmospheric pressure (Pa)
 * @returns Water density (kg/m³)
 */
export function densityH2o(tc: np.Array, patm: np.Array): np.Array {
  // lambda (bar·cm³/g)
  using tc2 = tc.mul(tc);
  using tc3 = tc2.mul(tc);
  using tc4 = tc3.mul(tc);
  using _jaxTmp1 = tc.mul(21.55053);
  using _jaxTmp2 = tc2.mul(0.4695911);
  using _jaxTmp3 = tc3.mul(3.096363e-3);
  using _jaxTmp4 = tc4.mul(7.341182e-6);
  using _lambda_0 = np.array(1788.316);
  using _lambda_1 = _lambda_0.add(_jaxTmp1);
  using _lambda_2 = _lambda_1.sub(_jaxTmp2);
  using _lambda_3 = _lambda_2.add(_jaxTmp3);
  using lambda_ = _lambda_3.sub(_jaxTmp4);

  // po (bar)
  using _jaxTmp5 = tc.mul(58.05267);
  using _jaxTmp6 = tc2.mul(1.1253317);
  using _jaxTmp7 = tc3.mul(6.6123869e-3);
  using _jaxTmp8 = tc4.mul(1.4661625e-5);
  using _po_0 = np.array(5918.499);
  using _po_1 = _po_0.add(_jaxTmp5);
  using _po_2 = _po_1.sub(_jaxTmp6);
  using _po_3 = _po_2.add(_jaxTmp7);
  using po = _po_3.sub(_jaxTmp8);

  // vinf (cm³/g)
  using tc5 = tc4.mul(tc);
  using tc6 = tc5.mul(tc);
  using tc7 = tc6.mul(tc);
  using tc8 = tc7.mul(tc);
  using tc9 = tc8.mul(tc);
  using _jaxTmp9 = tc.mul(7.435626e-4);
  using _jaxTmp10 = tc2.mul(3.704258e-5);
  using _jaxTmp11 = tc3.mul(6.315724e-7);
  using _jaxTmp12 = tc4.mul(9.829576e-9);
  using _jaxTmp13 = tc5.mul(1.197269e-10);
  using _jaxTmp14 = tc6.mul(1.005461e-12);
  using _jaxTmp15 = tc7.mul(5.437898e-15);
  using _jaxTmp16 = tc8.mul(1.69946e-17);
  using _jaxTmp17 = tc9.mul(2.295063e-20);
  using _vinf_0 = np.array(0.6980547);
  using _vinf_1 = _vinf_0.sub(_jaxTmp9);
  using _vinf_2 = _vinf_1.add(_jaxTmp10);
  using _vinf_3 = _vinf_2.sub(_jaxTmp11);
  using _vinf_4 = _vinf_3.add(_jaxTmp12);
  using _vinf_5 = _vinf_4.sub(_jaxTmp13);
  using _vinf_6 = _vinf_5.add(_jaxTmp14);
  using _vinf_7 = _vinf_6.sub(_jaxTmp15);
  using _vinf_8 = _vinf_7.add(_jaxTmp16);
  using vinf = _vinf_8.sub(_jaxTmp17);

  // pbar = patm * 1e-5
  using pbar = patm.mul(1e-5);

  // v = vinf + lambda / (po + pbar)
  using denom = po.add(pbar);
  using _jaxTmp18 = lambda_.div(denom);
  using v = vinf.add(_jaxTmp18);

  // density = 1e3 / v
  using _npConst1 = np.array(1e3);
  return _npConst1.div(v);
}
