import { np } from "../precision.js";

/** SpaFHy canopy/aerodynamic parameters. */
export interface SpafhyAeroParams {
  hc: np.Array; // Canopy height (m)
  zmeas: np.Array; // Measurement height above canopy (m)
  zground: np.Array; // Ground height (m)
  zo_ground: np.Array; // Ground roughness length (m)
  w_leaf: np.Array; // Leaf width (m)
}

export interface AeroResult {
  ra: np.Array; // Canopy aerodynamic resistance (s/m)
  rb: np.Array; // Canopy boundary layer resistance (s/m)
  ras: np.Array; // Ground aerodynamic resistance (s/m)
  ustar: np.Array; // Friction velocity (m/s)
  Uh: np.Array; // Wind speed at canopy top (m/s)
  Ug: np.Array; // Wind speed at ground (m/s)
}

/**
 * Aerodynamic resistances for canopy and soil layers.
 *
 * Fortran: `aerodynamics` in water_mod.f90
 * References: Cammalleri et al. (2010), Massman (1987), Magnani et al. (1998)
 *
 * @param LAI - Leaf area index (m²/m²)
 * @param Uo - Mean wind speed at reference height (m/s)
 * @param params - Canopy/aerodynamic parameters
 * @returns Aerodynamic resistances and velocities
 */
export function aerodynamics(
  LAI: np.Array,
  Uo: np.Array,
  params: SpafhyAeroParams,
): AeroResult {
  const kv = 0.4; // von Karman constant
  const beta_aero = 285.0;
  const eps = 1e-16;

  using zm1 = params.hc.add(params.zmeas);
  using _jaxTmp1 = params.hc.mul(0.1);
  using zg1 = np.minimum(params.zground, _jaxTmp1);
  using alpha1 = LAI.div(2); // wind attenuation coefficient
  using d = params.hc.mul(0.66); // displacement height
  using zom = params.hc.mul(0.123); // roughness length for momentum
  using zov = zom.mul(0.1); // scalar roughness length
  using zosv = params.zo_ground.mul(0.1);

  // ustar = Uo * kv / ln((zm1 - d) / zom)
  using zm1_d = zm1.sub(d);
  using _jaxTmp2 = zm1_d.div(zom);
  using _jaxTmp3 = np.log(_jaxTmp2);
  using _ustarMul = Uo.mul(kv);
  using ustarVal = _ustarMul.div(_jaxTmp3);
  const ustar = ustarVal.ref;

  // Uh = ustar / kv * ln((hc - d) / zom)
  using hc_d = params.hc.sub(d);
  using _jaxTmp4 = hc_d.div(zom);
  using _jaxTmp5 = np.log(_jaxTmp4);
  using _UhDiv = ustar.div(kv);
  using UhVal = _UhDiv.mul(_jaxTmp5);
  const Uh = UhVal.ref;

  // Ug = Uh * exp(alpha1 * (zn - 1))
  using _jaxTmp6 = zg1.div(params.hc);
  using zn = np.minimum(_jaxTmp6, 1.0);
  using _jaxTmp8 = zn.sub(1);
  using _jaxTmp9 = alpha1.mul(_jaxTmp8);
  using _jaxTmp10 = np.exp(_jaxTmp9);
  using UgVal = Uh.mul(_jaxTmp10);
  const Ug = UgVal.ref;

  // ra = 1/(kv² * Uo) * ln((zm1-d)/zom) * ln((zm1-d)/zov)
  using _jaxTmp11 = zm1_d.div(zom);
  using log_zom = np.log(_jaxTmp11);
  using _jaxTmp12 = zm1_d.div(zov);
  using log_zov = np.log(_jaxTmp12);
  using _jaxTmp13 = Uo.mul(kv * kv);
  using _raZero = Uo.mul(0.0);
  using _raOne = _raZero.add(1.0);
  using _raB1 = _raOne.div(_jaxTmp13);
  using _raB2 = _raB1.mul(log_zom);
  using raBase = _raB2.mul(log_zov);

  // rb = (1/LAI) * beta * sqrt((w_leaf / Uh) * (alpha1 / (1 - exp(-alpha1/2))))^0.5
  using halfAlpha = alpha1.mul(-0.5);
  using _jaxTmp14 = np.exp(halfAlpha);
  using _negExpHalfAlpha = _jaxTmp14.neg();
  using denomRb = _negExpHalfAlpha.add(1.0);
  using _jaxTmp15 = denomRb.add(eps);
  using _jaxTmp16 = alpha1.div(_jaxTmp15);
  using _rbWidthOverUh = params.w_leaf.div(Uh);
  using rbInner = _rbWidthOverUh.mul(_jaxTmp16);
  using _jaxTmp17 = LAI.add(eps);
  using _jaxTmp18 = np.sqrt(rbInner);
  using _rbZero = _jaxTmp17.mul(0.0);
  using _rbOne = _rbZero.add(1.0);
  using _rbV1 = _rbOne.div(_jaxTmp17);
  using _rbV2 = _rbV1.mul(beta_aero);
  using rb_val = _rbV2.mul(_jaxTmp18);

  // When LAI > eps, use rb_val; otherwise 0 (handled by LAI+eps above)
  const rb = rb_val.ref; // zero-copy refcount increment

  using raVal = raBase.add(rb);
  const ra = raVal.ref;

  // ras = 1/(kv² * Ug) * ln(zground/zo_ground) * ln(zground/zosv)
  using _jaxTmp19 = params.zground.div(params.zo_ground);
  using log_zg = np.log(_jaxTmp19);
  using _jaxTmp20 = params.zground.div(zosv);
  using log_zgsv = np.log(_jaxTmp20);
  using _jaxTmp21 = Ug.mul(kv * kv);
  using _rasZero = Ug.mul(0.0);
  using _rasOne = _rasZero.add(1.0);
  using _rasB1 = _rasOne.div(_jaxTmp21);
  using _rasB2 = _rasB1.mul(log_zg);
  using rasVal = _rasB2.mul(log_zgsv);
  const ras = rasVal.ref;

  return { ra, rb, ras, ustar, Uh, Ug };
}
