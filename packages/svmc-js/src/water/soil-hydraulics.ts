import { np } from "../precision.js";

/** Soil hydraulic parameters (Van Genuchten). */
export interface SoilHydroParams {
  nVan: np.Array; // Van Genuchten n parameter
  alphaVan: np.Array; // Van Genuchten α (1/kPa)
  watsat: np.Array; // Saturated water content (v/v)
  watres: np.Array; // Residual water content (v/v)
  ksat: np.Array; // Saturated hydraulic conductivity (m/s)
}

/**
 * Soil water retention curve: vol. water content → soil water potential (MPa).
 *
 * Fortran: `soil_water_retention_curve` in water_mod.f90
 * Uses the Van Genuchten model.
 *
 * @param volLiq - Volumetric liquid water content (v/v)
 * @param params - Soil hydraulic parameters
 * @returns Soil matric potential (MPa, negative)
 */
export function soilWaterRetentionCurve(
  volLiq: np.Array,
  params: SoilHydroParams,
): np.Array {
  using _one1 = np.array(1);
  using _jaxTmp1 = _one1.div(params.nVan);
  using _one2 = np.array(1);
  using m1 = _one2.sub(_jaxTmp1);
  using _jaxTmp2 = np.array(0.01);
  using effPorosity = np.maximum(_jaxTmp2, params.watsat);

  // satfrac = (vol_liq - watres) / (eff_porosity - watres)
  using num = volLiq.sub(params.watres);
  using den = effPorosity.sub(params.watres);
  using satfrac = num.div(den);

  // smp = -(1/alpha) * (satfrac^(1/(-m1)) - 1)^(1/n1) * 0.001
  using negM = m1.mul(-1);
  using _one3 = np.array(1);
  using _jaxTmp3 = _one3.div(negM);
  using _powResult = np.power(satfrac, _jaxTmp3);
  using inner = _powResult.sub(1);
  using _one4 = np.array(1);
  using _jaxTmp4 = _one4.div(params.nVan);
  using _jaxTmp5 = np.power(inner, _jaxTmp4);
  using _one5 = np.array(1);
  using _invAlpha = _one5.div(params.alphaVan);
  using _negInvAlpha = _invAlpha.mul(-1);
  using smp_kpa = _negInvAlpha.mul(_jaxTmp5);
  return smp_kpa.mul(0.001); // kPa → MPa
}

/**
 * Soil hydraulic conductivity (m/s) — Van Genuchten–Mualem.
 *
 * Fortran: `soil_hydraulic_conductivity` in water_mod.f90
 *
 * @param volLiq - Volumetric liquid water content (v/v)
 * @param params - Soil hydraulic parameters
 * @returns Hydraulic conductivity (m/s)
 */
export function soilHydraulicConductivity(
  volLiq: np.Array,
  params: SoilHydroParams,
): np.Array {
  using _npConst1 = np.array(1);
  using _jaxTmp1 = _npConst1.div(params.nVan);
  using _npConst2 = np.array(1);
  using m1 = _npConst2.sub(_jaxTmp1);
  using _jaxTmp2 = np.array(0.01);
  using effPorosity = np.maximum(_jaxTmp2, params.watsat);
  using _jaxTmp3 = effPorosity.sub(params.watres);
  using satfrac = volLiq.sub(params.watres).div(_jaxTmp3);

  // khydr = ksat * satfrac^0.5 * (1 - (1 - satfrac^(1/m1))^m1)²
  using sf_half = np.sqrt(satfrac);
  using _npConst3 = np.array(1);
  using _jaxTmp4 = _npConst3.div(m1);
  using inner = np.power(satfrac, _jaxTmp4);
  using _npConst4 = np.array(1);
  using _jaxTmp5 = _npConst4.sub(inner);
  using _jaxTmp6 = np.power(_jaxTmp5, m1);
  using _npConst5 = np.array(1);
  using bracket = _npConst5.sub(_jaxTmp6);
  using bracket2 = bracket.mul(bracket);
  using _ksat_sf = params.ksat.mul(sf_half);
  return _ksat_sf.mul(bracket2);
}
