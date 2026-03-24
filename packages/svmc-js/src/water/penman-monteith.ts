import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

/**
 * Saturation vapor pressure, slope, and psychrometric constant.
 *
 * Fortran: `e_sat` in water_mod.f90
 *
 * @param T - Air temperature (°C)
 * @param P - Ambient pressure (Pa)
 * @returns { esat, s, gamma } — saturation VP (Pa), slope (Pa/K), psychrometric const (Pa/K)
 */
export function eSat(
  T: np.Array,
  P: np.Array,
): { esat: np.Array; s: np.Array; gamma: np.Array } {
  const NT = 273.15;
  const cp = 1004.67; // J/kg/K

  using lambda_ = T.add(NT).mul(-2.37).add(3147.5).mul(1e3); // latent heat (J/kg)
  // esat = 1e3 * 0.6112 * exp(17.67*T / (T + 273.16 - 29.66))
  using _jaxTmp1 = T.add(273.16 - 29.66);
  using exArg = T.mul(17.67).div(_jaxTmp1);
  const esat = np.exp(exArg).mul(0.6112 * 1e3);

  // s = 17.502 * 240.97 * esat / (240.97 + T)²
  using denomS = T.add(240.97);
  using denomS2 = denomS.mul(denomS);
  const s = esat.mul(17.502 * 240.97).div(denomS2);

  // gamma = P * cp / (0.622 * lambda)
  using _jaxTmp2 = lambda_.mul(0.622);
  const gamma = P.mul(cp).div(_jaxTmp2);

  return { esat, s, gamma };
}

/**
 * Penman-Monteith latent heat flux (W/m²).
 *
 * Fortran: `penman_monteith` in water_mod.f90
 *
 * @param AE - Available energy (W/m²)
 * @param D - Vapor pressure deficit (Pa)
 * @param T - Temperature (°C)
 * @param Gs - Surface conductance (m/s)
 * @param Ga - Aerodynamic conductance (m/s)
 * @param P - Pressure (Pa)
 * @returns Latent heat flux LE (W/m²), clamped ≥ 0
 */
export function penmanMonteith(
  AE: np.Array,
  D: np.Array,
  T: np.Array,
  Gs: np.Array,
  Ga: np.Array,
  P: np.Array,
): np.Array {
  const cp = 1004.67;
  const rho = 1.25; // kg/m³

  const { esat, s, gamma: g } = eSat(T, P);
  esat.dispose();

  // PM = (s*AE + rho*cp*Ga*D) / (s + g*(1 + Ga/Gs))
  using _npConst1 = np.array(rho * cp);
  using _jaxTmp1 = _npConst1.mul(Ga).mul(D);
  using num = s.mul(AE).add(_jaxTmp1);
  using gaGs = Ga.div(Gs);
  using _jaxTmp2 = gaGs.add(1);
  using _jaxTmp3 = g.mul(_jaxTmp2);
  g.dispose();
  using den = s.add(_jaxTmp3);
  s.dispose();
  using result = num.div(den);
  using _jaxTmp4 = np.array(0);
  return np.maximum(result, _jaxTmp4);
}
