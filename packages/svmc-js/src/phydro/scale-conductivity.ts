import { np } from "../precision.js";

/** Environmental parameters for P-Hydro calculations. */
export interface ParEnv {
  viscosityWater: np.Array;
  densityWater: np.Array;
  patm: np.Array;
  tc: np.Array;
  vpd: np.Array;
}

/** Plant hydraulic parameters. */
export interface ParPlant {
  conductivity: np.Array; // Leaf conductivity (m)
  psi50: np.Array; // Water potential at 50% loss of conductivity (MPa)
  b: np.Array; // Shape parameter for vulnerability curve
}

/** Cost parameters for profit/least-cost optimization. */
export interface ParCost {
  alpha: np.Array; // Cost of Jmax
  gamma: np.Array; // Cost of hydraulic repair
}

/** Photosynthesis parameters. */
export interface ParPhotosynth {
  kmm: np.Array; // Michaelis-Menten coefficient (Pa)
  gammastar: np.Array; // CO2 compensation point (Pa)
  phi0: np.Array; // Quantum yield efficiency
  Iabs: np.Array; // Absorbed photosynthetically active radiation
  ca: np.Array; // Ambient CO2 partial pressure (Pa)
  patm: np.Array; // Atmospheric pressure (Pa)
  delta: np.Array; // Dark respiration fraction
}

/**
 * Scale plant hydraulic conductivity to mol/m²/s/MPa.
 *
 * Fortran: `scale_conductivity` in phydro_mod.f90
 */
export function scaleConduct(K: np.Array, parEnv: ParEnv): np.Array {
  const mol_h2o_per_kg_h2o = 55.5;
  // K2 = K / viscosity
  using K2 = K.div(parEnv.viscosityWater);
  // K3 = K2 * density * 55.5
  using _K2d = K2.mul(parEnv.densityWater);
  using K3 = _K2d.mul(mol_h2o_per_kg_h2o);
  // mol/m²/s/MPa
  return K3.mul(1e6);
}
