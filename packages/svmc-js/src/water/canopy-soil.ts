/**
 * Phase 3 SpaFHy composition functions: canopy water balance and soil water.
 *
 * Port of canopy_water_snow, ground_evaporation, canopy_water_flux, and
 * soil_water from water_mod.f90 in the Fortran SVMC model.
 */
import { np } from "../precision.js";
import { penmanMonteith } from "./penman-monteith.js";
import { aerodynamics } from "./aerodynamics.js";
import type { SpafhyAeroParams } from "./aerodynamics.js";
import {
  soilWaterRetentionCurve,
  soilHydraulicConductivity,
} from "./soil-hydraulics.js";
import type { SoilHydroParams } from "./soil-hydraulics.js";

// ── State & parameter interfaces ─────────────────────────────────────

export interface CanopyWaterState {
  CanopyStorage: np.Array;
  SWE: np.Array;
  swe_i: np.Array;
  swe_l: np.Array;
}

export interface CanopySnowParams {
  wmax: np.Array;
  wmaxsnow: np.Array;
  kmelt: np.Array;
  kfreeze: np.Array;
  fracSnowliq: np.Array;
  gsoil: np.Array;
}

export interface CanopySnowFlux {
  Throughfall: np.Array;
  Interception: np.Array;
  CanopyEvap: np.Array;
  Unloading: np.Array;
  PotInfiltration: np.Array;
  Melt: np.Array;
  Freeze: np.Array;
  mbe: np.Array;
}

export interface CanopyWaterFlux {
  Throughfall: np.Array;
  Interception: np.Array;
  CanopyEvap: np.Array;
  Unloading: np.Array;
  SoilEvap: np.Array;
  ET: np.Array;
  Transpiration: np.Array;
  PotInfiltration: np.Array;
  Melt: np.Array;
  Freeze: np.Array;
  mbe: np.Array;
}

export interface SoilWaterState {
  WatSto: np.Array;
  PondSto: np.Array;
  MaxWatSto: np.Array;
  MaxPondSto: np.Array;
  FcSto: np.Array;
  Wliq: np.Array;
  Psi: np.Array;
  Sat: np.Array;
  Kh: np.Array;
  beta: np.Array;
}

export interface SoilWaterFlux {
  Infiltration: np.Array;
  Runoff: np.Array;
  Drainage: np.Array;
  LateralFlow: np.Array;
  ET: np.Array;
  mbe: np.Array;
}

export interface SoilWaterResult {
  state: SoilWaterState;
  flux: SoilWaterFlux;
  trOut: np.Array;
  evapOut: np.Array;
  latflowOut: np.Array;
}

// ── Functions ────────────────────────────────────────────────────────

/**
 * Evaporation from soil surface (mm).
 *
 * Fortran: ground_evaporation in water_mod.f90
 */
export function groundEvaporation(
  tc: np.Array,
  ae: np.Array,
  vpd: np.Array,
  ras: np.Array,
  patm: np.Array,
  swe: np.Array,
  beta: np.Array,
  watSto: np.Array,
  gsoil: np.Array,
  timeStep: np.Array,
): np.Array {
  const eps = 1e-16;

  // lv = 1e3 * (3147.5 - 2.37 * (tc + 273.15))
  using _tcK = tc.add(273.15);
  using _lv1 = _tcK.mul(-2.37);
  using _lv2 = _lv1.add(3147.5);
  using lv = _lv2.mul(1e3);

  // ga_s = 1/ras
  using _one = np.array(1);
  using gaS = _one.div(ras);

  // erate = timeStep*3600 * beta * PM(...) / lv
  using dt = timeStep.mul(3600);
  using pm = penmanMonteith(ae, vpd, tc, gsoil, gaS, patm);
  using _dtBeta = dt.mul(beta);
  using _dtBetaPm = _dtBeta.mul(pm);
  using erate = _dtBetaPm.div(lv);

  // Cap at available water
  using soilEvapPre = np.minimum(watSto, erate);

  // PORT-BRANCH: water.ground_evaporation.snow_floor_zero
  using snowPresent = swe.greater(eps);
  return np.where(snowPresent, 0.0, soilEvapPre);
}

/**
 * Canopy interception, throughfall and snowpack dynamics.
 *
 * Fortran: canopy_water_snow in water_mod.f90
 */
export function canopyWaterSnow(
  state: CanopyWaterState,
  params: CanopySnowParams,
  tc: np.Array,
  pre: np.Array,
  ae: np.Array,
  d: np.Array,
  ra: np.Array,
  u: np.Array,
  lai: np.Array,
  patm: np.Array,
  timeStep: np.Array,
): [CanopyWaterState, CanopySnowFlux] {
  const eps = 1e-16;
  const tmin = 0.0;
  const tmax = 1.0;
  const tmelt = 0.0;

  using dt = timeStep.mul(3600);

  // PORT-BRANCH: water.canopy_water_snow.precip_phase
  using _fwSub = tc.sub(tmin);
  using _fwRaw = _fwSub.div(tmax - tmin);
  using fw = np.clip(_fwRaw, 0, 1);
  using _one1 = np.array(1);
  using fs = _one1.sub(fw);

  // Canopy storage capacities (mm)
  using wmaxTot = params.wmax.mul(lai);
  using wmaxsnowTot = params.wmaxsnow.mul(lai);

  // Latent heats (J/kg)
  using _tcK = tc.add(273.15);
  using _lv1 = _tcK.mul(-2.37);
  using _lv2 = _lv1.add(3147.5);
  using lv = _lv2.mul(1e3);
  using ls = lv.add(3.3e5);

  // Accumulated precipitation (mm)
  using prec = pre.mul(dt);

  // Aerodynamic conductance
  using _oneGa = np.array(1);
  using ga = _oneGa.div(ra);

  // ── Evaporation/sublimation from canopy ──

  // Guard wmaxsnowTot for LAI=0
  using wmaxsnowSafe = np.maximum(wmaxsnowTot, eps);
  using _wEps = state.CanopyStorage.add(eps);
  using _ceRatio = _wEps.div(wmaxsnowSafe);
  using _cePow = np.power(_ceRatio, -0.4);
  using ce = _cePow.mul(0.01);

  using _sqrtU = np.sqrt(u);
  using _shMul = _sqrtU.mul(3.0);
  using sh = _shMul.add(1.79);
  using _giMul1 = sh.mul(state.CanopyStorage);
  using _giProd = _giMul1.mul(ce);
  using _giDiv = _giProd.div(7.68);
  using gi = _giDiv.add(eps);

  // Sublimation rate (when no precip and T <= Tmin)
  using _dtDivLs = dt.div(ls);
  using pmSublim = penmanMonteith(ae, d, tc, gi, ga, patm);
  using erateSublim = _dtDivLs.mul(pmSublim);

  // Evaporation rate (when no precip and T > Tmin)
  using gsFree = np.array(1e6);
  using _dtDivLv = dt.div(lv);
  using pmEvap = penmanMonteith(ae, d, tc, gsFree, ga, patm);
  using erateEvap = _dtDivLv.mul(pmEvap);

  // PORT-BRANCH: water.canopy_water_snow.sublim_vs_evap
  using noPrecip = prec.lessEqual(0);
  using cold = tc.lessEqual(tmin);
  using warm = tc.greater(tmin);
  using condSublim = noPrecip.mul(cold);
  using condEvap = noPrecip.mul(warm);
  using _innerErate = np.where(condEvap, erateEvap, 0.0);
  using erateBase = np.where(condSublim, erateSublim, _innerErate);

  // PORT-BRANCH: water.canopy_water_snow.lai_evap_guard
  using laiPos = lai.greater(eps);
  using erate = np.where(laiPos, erateBase, 0.0);

  // PORT-BRANCH: water.canopy_water_snow.snow_unloading
  using tcGeTmin = tc.greaterEqual(tmin);
  using _excessW = state.CanopyStorage.sub(wmaxTot);
  using _excessClamped = np.maximum(_excessW, 0);
  const Unloading = np.where(tcGeTmin, _excessClamped, 0.0);

  // w after unloading
  using w1 = state.CanopyStorage.sub(Unloading);

  // PORT-BRANCH: water.canopy_water_snow.interception_phase
  // Snow interception (T < Tmin)
  using _precDivSnow = prec.div(wmaxsnowSafe);
  using _negPrecDivSnow = _precDivSnow.mul(-1);
  using _expSnow = np.exp(_negPrecDivSnow);
  using _one2 = np.array(1);
  using _oneMexpSnow = _one2.sub(_expSnow);
  using _snowCap = wmaxsnowTot.sub(w1);
  using intercSnowRaw = _snowCap.mul(_oneMexpSnow);
  using intercSnow = np.where(laiPos, intercSnowRaw, 0.0);

  // Rain interception (T >= Tmin)
  using wmaxSafe = np.maximum(wmaxTot, eps);
  using _precDivRain = prec.div(wmaxSafe);
  using _negPrecDivRain = _precDivRain.mul(-1);
  using _expRain = np.exp(_negPrecDivRain);
  using _one3 = np.array(1);
  using _oneMexpRain = _one3.sub(_expRain);
  using _rainCap = wmaxTot.sub(w1);
  using _rainCapPos = np.maximum(_rainCap, 0);
  using intercRainRaw = _rainCapPos.mul(_oneMexpRain);
  using intercRain = np.where(laiPos, intercRainRaw, 0.0);

  using tcLtTmin = tc.less(tmin);
  const Interception = np.where(tcLtTmin, intercSnow, intercRain);

  // Update w after interception
  using w2 = w1.add(Interception);

  // Evaporate from canopy
  using _w2eps = w2.add(eps);
  const CanopyEvap = np.minimum(erate, _w2eps);
  const w3 = w2.sub(CanopyEvap);

  // Throughfall
  using _trfA = prec.add(Unloading);
  const Throughfall = _trfA.sub(Interception);

  // PORT-BRANCH: water.canopy_water_snow.melt_freeze
  using _meltRate = params.kmelt.mul(dt);
  using _tcMtmelt = tc.sub(tmelt);
  using _meltAmt = _meltRate.mul(_tcMtmelt);
  using _meltCapped = np.minimum(state.swe_i, _meltAmt);
  const Melt = np.where(tcGeTmin, _meltCapped, 0.0);

  using tcLtMelt = tc.less(tmelt);
  using swelPos = state.swe_l.greater(0);
  using condFreeze = tcLtMelt.mul(swelPos);
  using _freezeRate = params.kfreeze.mul(dt);
  using _tmeltArr = np.array(tmelt);
  using _tmeltMtc = _tmeltArr.sub(tc);
  using _freezeAmt = _freezeRate.mul(_tmeltMtc);
  using _freezeCapped = np.minimum(state.swe_l, _freezeAmt);
  const Freeze = np.where(condFreeze, _freezeCapped, 0.0);

  // Snow ice and liquid
  using _fsTr = fs.mul(Throughfall);
  using _siceA = state.swe_i.add(_fsTr);
  using _siceB = _siceA.add(Freeze);
  using _siceC = _siceB.sub(Melt);
  const sice = np.maximum(_siceC, 0);

  using _fwTr = fw.mul(Throughfall);
  using _sliqA = state.swe_l.add(_fwTr);
  using _sliqB = _sliqA.sub(Freeze);
  using _sliqC = _sliqB.add(Melt);
  using sliqPre = np.maximum(_sliqC, 0);

  // Potential infiltration — excess liquid drains from snowpack
  using _siceFrac = sice.mul(params.fracSnowliq);
  using _potInfRaw = sliqPre.sub(_siceFrac);
  const PotInfiltration = np.maximum(_potInfRaw, 0);

  using _sliqFinal = sliqPre.sub(PotInfiltration);
  const sliq = np.maximum(_sliqFinal, 0);

  // New state
  const SWE = sliq.add(sice);
  const newState: CanopyWaterState = {
    CanopyStorage: w3,
    SWE,
    swe_i: sice,
    swe_l: sliq,
  };

  // Mass-balance error
  using _wSwe = w3.add(SWE);
  using _woSweo = state.CanopyStorage.add(state.SWE);
  using _mbeLeft = _wSwe.sub(_woSweo);
  using _mbePrec = prec.sub(CanopyEvap);
  using _mbeFlux = _mbePrec.sub(PotInfiltration);
  const mbe = _mbeLeft.sub(_mbeFlux);

  const flux: CanopySnowFlux = {
    Throughfall,
    Interception,
    CanopyEvap,
    Unloading,
    PotInfiltration,
    Melt,
    Freeze,
    mbe,
  };

  return [newState, flux];
}

/**
 * Canopy water balance orchestrator.
 *
 * Fortran: canopy_water_flux in water_mod.f90
 * Calls aerodynamics → canopy_water_snow → ground_evaporation.
 */
export function canopyWaterFlux(
  rn: np.Array,
  ta: np.Array,
  prec: np.Array,
  vpd: np.Array,
  u: np.Array,
  patm: np.Array,
  fapar: np.Array,
  lai: np.Array,
  cwState: CanopyWaterState,
  swBeta: np.Array,
  swWatSto: np.Array,
  aeroParams: SpafhyAeroParams,
  csParams: CanopySnowParams,
  timeStep: np.Array,
): [CanopyWaterState, CanopyWaterFlux] {
  // Aerodynamic resistances
  const aero = aerodynamics(lai, u, aeroParams);
  using ra = aero.ra.ref;
  using ras = aero.ras.ref;
  aero.ra.dispose();
  aero.ras.dispose();
  aero.rb.dispose();
  aero.ustar.dispose();
  aero.Uh.dispose();
  aero.Ug.dispose();

  // Canopy interception & snowpack
  using aeCanopy = rn.mul(fapar);
  const [newState, csFlux] = canopyWaterSnow(
    cwState, csParams, ta, prec, aeCanopy, vpd, ra, u, lai, patm, timeStep,
  );

  // Ground evaporation
  using _negFapar = fapar.neg();
  using _oneMinusFapar = _negFapar.add(1.0);
  using aeSoil = rn.mul(_oneMinusFapar);
  const SoilEvap = groundEvaporation(
    ta, aeSoil, vpd, ras, patm, newState.SWE, swBeta, swWatSto,
    csParams.gsoil, timeStep,
  );

  const ET = SoilEvap.mul(0.0);
  const Transpiration = SoilEvap.mul(0.0);

  const flux: CanopyWaterFlux = {
    Throughfall: csFlux.Throughfall,
    Interception: csFlux.Interception,
    CanopyEvap: csFlux.CanopyEvap,
    Unloading: csFlux.Unloading,
    SoilEvap,
    ET,
    Transpiration,
    PotInfiltration: csFlux.PotInfiltration,
    Melt: csFlux.Melt,
    Freeze: csFlux.Freeze,
    mbe: csFlux.mbe,
  };

  return [newState, flux];
}

/**
 * Soil water balance in 1-layer bucket.
 *
 * Fortran: soil_water in water_mod.f90
 */
export function soilWater(
  state: SoilWaterState,
  soilParams: SoilHydroParams,
  maxPoros: np.Array,
  potinf: np.Array,
  tr: np.Array,
  evap: np.Array,
  _latflow: np.Array,
  timeStep: np.Array,
): SoilWaterResult {
  const eps = 1e-16;

  using dtS = timeStep.mul(3600);

  // rr = potinf + PondSto
  using rr = potinf.add(state.PondSto);

  // Clamp transpiration and evaporation to available water
  using _trA1 = state.WatSto.add(rr);
  using _trAvail = _trA1.sub(eps);
  const trOut = np.minimum(tr, _trAvail);

  using _evA1 = state.WatSto.add(rr);
  using _evA2 = _evA1.sub(trOut);
  using _evAvail = _evA2.sub(eps);
  const evapOut = np.minimum(evap, _evAvail);

  // Water storage after upward fluxes
  using _ws1 = state.WatSto.sub(trOut);
  using watSto0 = _ws1.sub(evapOut);

  // Vertical drainage (limited to surplus above field capacity)
  using _drainMul1 = state.Kh.mul(dtS);
  using _drainRaw = _drainMul1.mul(1000);
  using _drainSurplus = watSto0.sub(state.FcSto);
  using _drainSurplusClamped = np.maximum(_drainSurplus, 0);
  const drain = np.minimum(_drainRaw, _drainSurplusClamped);

  // Lateral drainage (hardcoded to 0 in Fortran)
  const latflowOut = np.array(0);

  // Infiltration limited by available water or storage space
  using _infilAdd1 = state.MaxWatSto.add(drain);
  using _infilCap = _infilAdd1.add(latflowOut);
  const infil = np.minimum(rr, _infilCap);

  // Update soil water storage
  using _wsAdd = watSto0.add(infil);
  using _wsSub1 = _wsAdd.sub(drain);
  const watSto = _wsSub1.sub(latflowOut);

  // Pond storage and runoff
  using _toPondRaw = rr.sub(infil);
  using toPond = np.maximum(_toPondRaw, 0);
  const PondSto = np.minimum(toPond, state.MaxPondSto);
  using _runoffRaw = toPond.sub(PondSto);
  const runoff = np.maximum(_runoffRaw, 0);

  // Derived state: volumetric water content
  using _satRatio = watSto.div(state.MaxWatSto);
  using _satCapped = np.minimum(_satRatio, 1);
  const Wliq = maxPoros.mul(_satCapped);
  const Sat = Wliq.div(maxPoros);
  const beta = np.minimum(Sat, 1);

  // Soil water potential and hydraulic conductivity
  const Psi = soilWaterRetentionCurve(Wliq, soilParams);
  const Kh = soilHydraulicConductivity(Wliq, soilParams);

  // Mass-balance error
  using _mbeWat = watSto.sub(state.WatSto);
  using _mbePond = PondSto.sub(state.PondSto);
  using _mbeSto = _mbeWat.add(_mbePond);
  using _mf1a = rr.sub(trOut);
  using _mf1 = _mf1a.sub(evapOut);
  using _mf2 = _mf1.sub(drain);
  using _mf3 = _mf2.sub(latflowOut);
  using _mbeFlux = _mf3.sub(runoff);
  const mbe = _mbeSto.sub(_mbeFlux);

  const newState: SoilWaterState = {
    WatSto: watSto,
    PondSto,
    MaxWatSto: state.MaxWatSto,
    MaxPondSto: state.MaxPondSto,
    FcSto: state.FcSto,
    Wliq,
    Psi,
    Sat,
    Kh,
    beta,
  };

  const flux: SoilWaterFlux = {
    Infiltration: infil,
    Runoff: runoff,
    Drainage: drain,
    LateralFlow: np.array(0),
    ET: trOut.add(evapOut),
    mbe,
  };

  return { state: newState, flux, trOut, evapOut, latflowOut };
}
