import { np } from "../precision.js";
import { densityH2o } from "./density-h2o.js";

/**
 * Viscosity of water (Pa·s) as a function of temperature and atmospheric pressure.
 *
 * Fortran: `viscosity_h2o` in phydro_mod.f90
 * References: Huber et al. (2009)
 *
 * @param tc - Temperature (°C)
 * @param patm - Atmospheric pressure (Pa)
 * @returns Viscosity (Pa·s)
 */
export function viscosityH2o(tc: np.Array, patm: np.Array): np.Array {
  const tk_ast = 647.096;
  const rho_ast = 322.0;
  const mu_ast = 1e-6;

  using rho = densityH2o(tc, patm);

  using _tbar_1 = tc.add(273.15);
  using tbar = _tbar_1.div(tk_ast);
  using rbar = rho.div(rho_ast);

  // mu0 (Eq. 11 & Table 2)
  using tbar2 = tbar.mul(tbar);
  using tbar3 = tbar2.mul(tbar);
  using _npConst1 = np.array(2.20462);
  using _jaxTmp1 = _npConst1.div(tbar);
  using _npConst2 = np.array(0.6366564);
  using _jaxTmp2 = _npConst2.div(tbar2);
  using _npConst3 = np.array(0.241605);
  using _jaxTmp3 = _npConst3.div(tbar3);
  using _mu0_d_0 = np.array(1.67752);
  using _mu0_d_1 = _mu0_d_0.add(_jaxTmp1);
  using _mu0_d_2 = _mu0_d_1.add(_jaxTmp2);
  using mu0_denom = _mu0_d_2.sub(_jaxTmp3);
  using _mu0_1 = np.sqrt(tbar);
  using _mu0_2 = _mu0_1.mul(1e2);
  using mu0 = _mu0_2.div(mu0_denom);

  // Table 3 coefficients, Huber et al. (2009)
  // h_array[j][i], j=0..6, i=0..5
  const h = [
    [0.520094, 0.0850895, -1.08374, -0.289555, 0.0, 0.0],
    [0.222531, 0.999115, 1.88797, 1.26613, 0.0, 0.120573],
    [-0.281378, -0.906851, -0.772479, -0.489837, -0.25704, 0.0],
    [0.161913, 0.257399, 0.0, 0.0, 0.0, 0.0],
    [-0.0325372, 0.0, 0.0, 0.0698452, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.00872102, 0.0],
    [0.0, 0.0, 0.0, -0.00435673, 0.0, -0.000593264],
  ];

  // mu1 (Eq. 12 & Table 3)
  // Compute as a scalar sum since the h coefficients are constants.
  // mu1 = exp(rbar * sum_ij)
  using _npConst4 = np.array(1);
  using _ctbar_1 = _npConst4.div(tbar);
  using ctbar = _ctbar_1.sub(1);
  using rbar_m1 = rbar.sub(1);

  // Build the double sum
  let sum_result = np.array(0);
  for (let i = 0; i < 6; i++) {
    using _jaxTmp4 = np.array(i);
    using ctbar_pow_i = i === 0 ? np.array(1) : np.power(ctbar, _jaxTmp4);
    let coef2 = np.array(0);
    for (let j = 0; j < 7; j++) {
      if (h[j][i] === 0.0) continue;
      using _jaxTmp5 = np.array(j);
      using rbar_pow_j =
        j === 0 ? np.array(1) : np.power(rbar_m1, _jaxTmp5);
      using term = rbar_pow_j.mul(h[j][i]);
      const newCoef2 = coef2.add(term);
      coef2.dispose();
      coef2 = newCoef2;
    }
    using prod = ctbar_pow_i.mul(coef2);
    const newSum = sum_result.add(prod);
    sum_result.dispose();
    sum_result = newSum;
    coef2.dispose();
  }

  using _jaxTmp6 = rbar.mul(sum_result);
  using mu1 = np.exp(_jaxTmp6);
  sum_result.dispose();

  // mu = mu0 * mu1 * mu_ast
  using _mu_partial = mu0.mul(mu1);
  return _mu_partial.mul(mu_ast);
}
