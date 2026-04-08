/**
 * Incremental reproduction: starting from the passing reduced scan test,
 * add real building blocks one by one to find which triggers the
 * "Referenced tracer Array has been disposed" error.
 *
 * Levels:
 *  0: Baseline (water only — already passes)
 *  1: + met_rolling / is_first_met carry fields
 *  2: + accumulator carry fields (temp_acc, precip_acc, gpp_acc, vcmax_acc, etc.)
 *  3: + met smoothing computation inside hourly body
 *  4: + pmodelHydraulicsNumerical inside hourly body
 *  5: + GPP / transpiration post-processing
 *  6: + daily post-processing (allocation, YASSO)
 */
import { describe, expect, it } from "vitest";
import { lax, tree, type JsTree } from "@hamk-uas/jax-js-nonconsuming";
import { np } from "../src/precision.js";
import { initializationSpafhy } from "../src/integration.js";
import {
  canopyWaterFlux,
  soilWater,
  type CanopySnowParams,
  type CanopyWaterState,
  type SoilWaterState,
  type SpafhyAeroParams,
} from "../src/water/index.js";
import type { SoilHydroParams } from "../src/water/soil-hydraulics.js";
import { pmodelHydraulicsNumerical, densityH2o } from "../src/phydro/index.js";
import qvidjaRef from "../../../website/public/qvidja-v1-reference.json";

const K_EXT = 0.5;
const TIME_STEP = 1.0;
const ALPHA_SMOOTH1 = 0.01;
const ALPHA_SMOOTH2 = 0.0016;
const LAI_GUARD = 1.0e-6;

function buildInputs() {
  const defaults = qvidjaRef.defaults;
  const hourly = qvidjaRef.hourly;
  const daily = qvidjaRef.daily;
  return {
    hourly_temp: hourly.temp_hr.slice(0, 24),
    hourly_rg: hourly.rg_hr.slice(0, 24),
    hourly_prec: hourly.prec_hr.slice(0, 24),
    hourly_vpd: hourly.vpd_hr.slice(0, 24),
    hourly_pres: hourly.pres_hr.slice(0, 24),
    hourly_co2: hourly.co2_hr.slice(0, 24),
    hourly_wind: hourly.wind_hr.slice(0, 24),
    daily_lai: daily.lai_day[0],
    soil_depth: defaults.soil_depth,
    max_poros: defaults.max_poros,
    fc: defaults.fc,
    maxpond: 0.0,
    watsat: defaults.max_poros,
    watres: 0.0,
    alpha_van: 5.92,
    n_van: 1.14,
    ksat: defaults.ksat,
    hc: 0.6,
    zmeas: 2.0,
    zground: 0.1,
    zo_ground: 0.01,
    w_leaf: 0.01,
    wmax: 0.5,
    wmaxsnow: 4.5,
    kmelt: 2.8934e-5,
    kfreeze: 5.79e-6,
    frac_snowliq: 0.05,
    gsoil: 5.0e-3,
    conductivity: defaults.conductivity,
    psi50: defaults.psi50,
    b_param: defaults.b,
    alpha_cost: defaults.alpha,
    gamma_cost: defaults.gamma,
    rdark: defaults.rdark,
  };
}

function buildParams(inputs: ReturnType<typeof buildInputs>) {
  const soilParams = tree.makeDisposable({
    watsat: np.array(inputs.watsat),
    watres: np.array(inputs.watres),
    alphaVan: np.array(inputs.alpha_van),
    nVan: np.array(inputs.n_van),
    ksat: np.array(inputs.ksat),
  }) as SoilHydroParams;
  const aeroParams = tree.makeDisposable({
    hc: np.array(inputs.hc),
    zmeas: np.array(inputs.zmeas),
    zground: np.array(inputs.zground),
    zo_ground: np.array(inputs.zo_ground),
    w_leaf: np.array(inputs.w_leaf),
  }) as SpafhyAeroParams;
  const csParams = tree.makeDisposable({
    wmax: np.array(inputs.wmax),
    wmaxsnow: np.array(inputs.wmaxsnow),
    kmelt: np.array(inputs.kmelt),
    kfreeze: np.array(inputs.kfreeze),
    fracSnowliq: np.array(inputs.frac_snowliq),
    gsoil: np.array(inputs.gsoil),
  }) as CanopySnowParams;
  return { soilParams, aeroParams, csParams };
}

describe("incremental scan building blocks", () => {
  // ----- Level 0: Baseline (water-only, already passes) ----- //
  it("L0: water-only baseline", () => {
    const inputs = buildInputs();
    const { soilParams, aeroParams, csParams } = buildParams(inputs);

    using maxPoros = np.array(inputs.max_poros);
    using lai = np.array(inputs.daily_lai);
    using faparArg = lai.mul(-K_EXT);
    using faparExp = np.exp(faparArg);
    using one = np.array(1.0);
    using fapar = one.sub(faparExp);
    using hourlyTemp = np.array(inputs.hourly_temp);
    using hourlyRg = np.array(inputs.hourly_rg);
    using hourlyPrec = np.array(inputs.hourly_prec);
    using hourlyVpd = np.array(inputs.hourly_vpd);
    using hourlyPres = np.array(inputs.hourly_pres);
    using hourlyWind = np.array(inputs.hourly_wind);
    using dayXs = np.array([0.0]);

    const [cwInit, swInit] = initializationSpafhy(
      inputs.soil_depth, inputs.max_poros, inputs.fc, inputs.maxpond, soilParams,
    );
    const timeStep = lai.mul(0.0).add(TIME_STEP);
    const zeroLatflow = lai.mul(0.0);

    type Carry = { cw_state: CanopyWaterState; sw_state: SoilWaterState; et_acc: np.Array };
    const dailyInit = { cw_state: cwInit, sw_state: swInit };
    const hourlyForcing = {
      temp_k: hourlyTemp, rg: hourlyRg, prec: hourlyPrec,
      vpd: hourlyVpd, pres: hourlyPres, wind: hourlyWind,
    };

    let finalCarry: typeof dailyInit | null = null;
    let ys: np.Array | null = null;

    try {
      const dailyBody = (
        carry: typeof dailyInit, _x: np.Array,
      ): [typeof dailyInit, np.Array] => {
        const hourlyInit: Carry = {
          cw_state: carry.cw_state,
          sw_state: carry.sw_state,
          et_acc: carry.cw_state.CanopyStorage.mul(0.0),
        };

        const hourlyBody = (hourIdx: np.Array, hc: Carry): Carry => {
          const tempK = lax.dynamicSlice(hourlyForcing.temp_k, [hourIdx], [1]).reshape([]);
          const rg = lax.dynamicSlice(hourlyForcing.rg, [hourIdx], [1]).reshape([]);
          const prec = lax.dynamicSlice(hourlyForcing.prec, [hourIdx], [1]).reshape([]);
          const vpd = lax.dynamicSlice(hourlyForcing.vpd, [hourIdx], [1]).reshape([]);
          const pres = lax.dynamicSlice(hourlyForcing.pres, [hourIdx], [1]).reshape([]);
          const wind = lax.dynamicSlice(hourlyForcing.wind, [hourIdx], [1]).reshape([]);
          const tc = tempK.sub(273.15);
          const trSpafhy = hc.sw_state.Wliq.mul(0.0);

          const [cwState, cwFlux] = canopyWaterFlux(
            rg.mul(0.7), tc, prec, vpd, wind, pres,
            fapar, lai, hc.cw_state, hc.sw_state.beta, hc.sw_state.WatSto,
            aeroParams, csParams, timeStep,
          );
          const soilResult = soilWater(
            hc.sw_state, soilParams, maxPoros,
            cwFlux.PotInfiltration, trSpafhy, cwFlux.SoilEvap,
            zeroLatflow, timeStep,
          );
          const newEtAcc = hc.et_acc.add(trSpafhy).add(cwFlux.SoilEvap).add(cwFlux.CanopyEvap);
          return { cw_state: cwState, sw_state: soilResult.state, et_acc: newEtAcc };
        };

        const finalHourly = lax.foriLoop(
          0, 24,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          hourlyInit as unknown as JsTree<np.Array>,
        ) as unknown as Carry;

        return [
          { cw_state: finalHourly.cw_state, sw_state: finalHourly.sw_state },
          finalHourly.et_acc,
        ];
      };

      [finalCarry, ys] = lax.scan(
        dailyBody as unknown as (c: JsTree<np.Array>, x: JsTree<np.Array>) => [JsTree<np.Array>, JsTree<np.Array>],
        dailyInit as unknown as JsTree<np.Array>,
        dayXs as unknown as JsTree<np.Array>,
        { length: 1 },
      ) as unknown as [typeof dailyInit, np.Array];

      expect(ys.shape[0]).toBe(1);
      expect(finalCarry.sw_state.Wliq.shape).toEqual([]);
    } finally {
      if (ys != null) ys.dispose();
      if (finalCarry != null) tree.dispose(finalCarry);
      tree.dispose(dailyInit);
      tree.dispose(soilParams);
      tree.dispose(aeroParams);
      tree.dispose(csParams);
    }
  });

  // ----- Level 1: + met_rolling + is_first_met carry ----- //
  it("L1: + met_rolling and is_first_met in carry", () => {
    const inputs = buildInputs();
    const { soilParams, aeroParams, csParams } = buildParams(inputs);

    using maxPoros = np.array(inputs.max_poros);
    using lai = np.array(inputs.daily_lai);
    using faparArg = lai.mul(-K_EXT);
    using faparExp = np.exp(faparArg);
    using one = np.array(1.0);
    using fapar = one.sub(faparExp);
    using hourlyTemp = np.array(inputs.hourly_temp);
    using hourlyRg = np.array(inputs.hourly_rg);
    using hourlyPrec = np.array(inputs.hourly_prec);
    using hourlyVpd = np.array(inputs.hourly_vpd);
    using hourlyPres = np.array(inputs.hourly_pres);
    using hourlyWind = np.array(inputs.hourly_wind);
    using dayXs = np.array([0.0]);

    const [cwInit, swInit] = initializationSpafhy(
      inputs.soil_depth, inputs.max_poros, inputs.fc, inputs.maxpond, soilParams,
    );
    const timeStep = lai.mul(0.0).add(TIME_STEP);
    const zeroLatflow = lai.mul(0.0);

    type Carry = {
      cw_state: CanopyWaterState;
      sw_state: SoilWaterState;
      met_rolling: np.Array;
      is_first_met: np.Array;
      et_acc: np.Array;
    };

    const dailyInit = {
      cw_state: cwInit,
      sw_state: swInit,
      met_rolling: np.array([0.0, 0.0]),
      is_first_met: np.array(true),
    };

    const hourlyForcing = {
      temp_k: hourlyTemp, rg: hourlyRg, prec: hourlyPrec,
      vpd: hourlyVpd, pres: hourlyPres, wind: hourlyWind,
    };

    let finalCarry: typeof dailyInit | null = null;
    let ys: np.Array | null = null;

    try {
      const dailyBody = (
        carry: typeof dailyInit, _x: np.Array,
      ): [typeof dailyInit, np.Array] => {
        const hourlyInit: Carry = {
          cw_state: carry.cw_state,
          sw_state: carry.sw_state,
          met_rolling: carry.met_rolling,
          is_first_met: carry.is_first_met,
          et_acc: carry.cw_state.CanopyStorage.mul(0.0),
        };

        const hourlyBody = (hourIdx: np.Array, hc: Carry): Carry => {
          const tempK = lax.dynamicSlice(hourlyForcing.temp_k, [hourIdx], [1]).reshape([]);
          const rg = lax.dynamicSlice(hourlyForcing.rg, [hourIdx], [1]).reshape([]);
          const prec = lax.dynamicSlice(hourlyForcing.prec, [hourIdx], [1]).reshape([]);
          const vpd = lax.dynamicSlice(hourlyForcing.vpd, [hourIdx], [1]).reshape([]);
          const pres = lax.dynamicSlice(hourlyForcing.pres, [hourIdx], [1]).reshape([]);
          const wind = lax.dynamicSlice(hourlyForcing.wind, [hourIdx], [1]).reshape([]);
          const tc = tempK.sub(273.15);
          const trSpafhy = hc.sw_state.Wliq.mul(0.0);

          const [cwState, cwFlux] = canopyWaterFlux(
            rg.mul(0.7), tc, prec, vpd, wind, pres,
            fapar, lai, hc.cw_state, hc.sw_state.beta, hc.sw_state.WatSto,
            aeroParams, csParams, timeStep,
          );
          const soilResult = soilWater(
            hc.sw_state, soilParams, maxPoros,
            cwFlux.PotInfiltration, trSpafhy, cwFlux.SoilEvap,
            zeroLatflow, timeStep,
          );
          const newEtAcc = hc.et_acc.add(trSpafhy).add(cwFlux.SoilEvap).add(cwFlux.CanopyEvap);

          // Met smoothing computation
          const metPrecMelt = cwFlux.Melt.div(TIME_STEP * 3600.0);
          const metPrec = prec.add(metPrecMelt);
          const metDaily = np.stack([tc, metPrec]);
          const smoothedTempOld = hc.met_rolling.slice(0);
          const smoothedPrecOld = hc.met_rolling.slice(1);
          const metDailyTemp = metDaily.slice(0);
          const metDailyPrec = metDaily.slice(1);
          const smoothTempNew = metDailyTemp.mul(ALPHA_SMOOTH1);
          const smoothTempOldScaled = smoothedTempOld.mul(1.0 - ALPHA_SMOOTH1);
          const smoothTemp = smoothTempNew.add(smoothTempOldScaled);
          const smoothPrecNew = metDailyPrec.mul(ALPHA_SMOOTH2);
          const smoothPrecOldScaled = smoothedPrecOld.mul(1.0 - ALPHA_SMOOTH2);
          const smoothPrec = smoothPrecNew.add(smoothPrecOldScaled);
          const smoothed = np.stack([smoothTemp, smoothPrec]);
          const newMetRolling = np.where(hc.is_first_met, metDaily, smoothed);
          const notFirstMet = np.where(hc.is_first_met, false, hc.is_first_met);

          return {
            cw_state: cwState,
            sw_state: soilResult.state,
            met_rolling: newMetRolling,
            is_first_met: notFirstMet,
            et_acc: newEtAcc,
          };
        };

        const finalHourly = lax.foriLoop(
          0, 24,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          hourlyInit as unknown as JsTree<np.Array>,
        ) as unknown as Carry;

        return [
          {
            cw_state: finalHourly.cw_state,
            sw_state: finalHourly.sw_state,
            met_rolling: finalHourly.met_rolling,
            is_first_met: finalHourly.is_first_met,
          },
          finalHourly.et_acc,
        ];
      };

      [finalCarry, ys] = lax.scan(
        dailyBody as unknown as (c: JsTree<np.Array>, x: JsTree<np.Array>) => [JsTree<np.Array>, JsTree<np.Array>],
        dailyInit as unknown as JsTree<np.Array>,
        dayXs as unknown as JsTree<np.Array>,
        { length: 1 },
      ) as unknown as [typeof dailyInit, np.Array];

      expect(ys.shape[0]).toBe(1);
      expect(finalCarry.sw_state.Wliq.shape).toEqual([]);
    } finally {
      if (ys != null) ys.dispose();
      if (finalCarry != null) tree.dispose(finalCarry);
      tree.dispose(dailyInit);
      tree.dispose(soilParams);
      tree.dispose(aeroParams);
      tree.dispose(csParams);
    }
  });

  // ----- Level 2: + accumulator carry fields ----- //
  it("L2: + temp/precip/gpp/vcmax accumulators in carry", () => {
    const inputs = buildInputs();
    const { soilParams, aeroParams, csParams } = buildParams(inputs);

    using maxPoros = np.array(inputs.max_poros);
    using lai = np.array(inputs.daily_lai);
    using faparArg = lai.mul(-K_EXT);
    using faparExp = np.exp(faparArg);
    using one = np.array(1.0);
    using fapar = one.sub(faparExp);
    using hourlyTemp = np.array(inputs.hourly_temp);
    using hourlyRg = np.array(inputs.hourly_rg);
    using hourlyPrec = np.array(inputs.hourly_prec);
    using hourlyVpd = np.array(inputs.hourly_vpd);
    using hourlyPres = np.array(inputs.hourly_pres);
    using hourlyWind = np.array(inputs.hourly_wind);
    using dayXs = np.array([0.0]);

    const [cwInit, swInit] = initializationSpafhy(
      inputs.soil_depth, inputs.max_poros, inputs.fc, inputs.maxpond, soilParams,
    );
    const timeStep = lai.mul(0.0).add(TIME_STEP);
    const zeroLatflow = lai.mul(0.0);

    type Carry = {
      cw_state: CanopyWaterState;
      sw_state: SoilWaterState;
      met_rolling: np.Array;
      is_first_met: np.Array;
      temp_acc: np.Array;
      precip_acc: np.Array;
      gpp_acc: np.Array;
      vcmax_acc: np.Array;
      num_gpp: np.Array;
      num_vcmax: np.Array;
      et_acc: np.Array;
    };

    const dailyInit = {
      cw_state: cwInit,
      sw_state: swInit,
      met_rolling: np.array([0.0, 0.0]),
      is_first_met: np.array(true),
    };

    const hourlyForcing = {
      temp_k: hourlyTemp, rg: hourlyRg, prec: hourlyPrec,
      vpd: hourlyVpd, pres: hourlyPres, wind: hourlyWind,
    };

    let finalCarry: typeof dailyInit | null = null;
    let ys: np.Array | null = null;

    try {
      const dailyBody = (
        carry: typeof dailyInit, _x: np.Array,
      ): [typeof dailyInit, np.Array] => {
        // Seed accumulators from carry scalars (same pattern as makeDailyStep)
        const zeroAcc = carry.cw_state.CanopyStorage.mul(0.0);
        const hourlyInit: Carry = {
          cw_state: carry.cw_state,
          sw_state: carry.sw_state,
          met_rolling: carry.met_rolling,
          is_first_met: carry.is_first_met,
          temp_acc: zeroAcc,
          precip_acc: carry.cw_state.CanopyStorage.mul(0.0),
          gpp_acc: carry.cw_state.CanopyStorage.mul(0.0),
          vcmax_acc: carry.cw_state.CanopyStorage.mul(0.0),
          num_gpp: carry.cw_state.CanopyStorage.mul(0.0),
          num_vcmax: carry.cw_state.CanopyStorage.mul(0.0),
          et_acc: carry.cw_state.CanopyStorage.mul(0.0),
        };

        const hourlyBody = (hourIdx: np.Array, hc: Carry): Carry => {
          const tempK = lax.dynamicSlice(hourlyForcing.temp_k, [hourIdx], [1]).reshape([]);
          const rg = lax.dynamicSlice(hourlyForcing.rg, [hourIdx], [1]).reshape([]);
          const prec = lax.dynamicSlice(hourlyForcing.prec, [hourIdx], [1]).reshape([]);
          const vpd = lax.dynamicSlice(hourlyForcing.vpd, [hourIdx], [1]).reshape([]);
          const pres = lax.dynamicSlice(hourlyForcing.pres, [hourIdx], [1]).reshape([]);
          const wind = lax.dynamicSlice(hourlyForcing.wind, [hourIdx], [1]).reshape([]);
          const tc = tempK.sub(273.15);
          const trSpafhy = hc.sw_state.Wliq.mul(0.0);

          const [cwState, cwFlux] = canopyWaterFlux(
            rg.mul(0.7), tc, prec, vpd, wind, pres,
            fapar, lai, hc.cw_state, hc.sw_state.beta, hc.sw_state.WatSto,
            aeroParams, csParams, timeStep,
          );
          const soilResult = soilWater(
            hc.sw_state, soilParams, maxPoros,
            cwFlux.PotInfiltration, trSpafhy, cwFlux.SoilEvap,
            zeroLatflow, timeStep,
          );
          const newEtAcc = hc.et_acc.add(trSpafhy).add(cwFlux.SoilEvap).add(cwFlux.CanopyEvap);

          // Met smoothing
          const metPrecMelt = cwFlux.Melt.div(TIME_STEP * 3600.0);
          const metPrec = prec.add(metPrecMelt);
          const metDaily = np.stack([tc, metPrec]);
          const smoothedTempOld = hc.met_rolling.slice(0);
          const smoothedPrecOld = hc.met_rolling.slice(1);
          const metDailyTemp = metDaily.slice(0);
          const metDailyPrec = metDaily.slice(1);
          const smoothTempNew = metDailyTemp.mul(ALPHA_SMOOTH1);
          const smoothTempOldScaled = smoothedTempOld.mul(1.0 - ALPHA_SMOOTH1);
          const smoothTemp = smoothTempNew.add(smoothTempOldScaled);
          const smoothPrecNew = metDailyPrec.mul(ALPHA_SMOOTH2);
          const smoothPrecOldScaled = smoothedPrecOld.mul(1.0 - ALPHA_SMOOTH2);
          const smoothPrec = smoothPrecNew.add(smoothPrecOldScaled);
          const smoothed = np.stack([smoothTemp, smoothPrec]);
          const newMetRolling = np.where(hc.is_first_met, metDaily, smoothed);
          const notFirstMet = np.where(hc.is_first_met, false, hc.is_first_met);

          // Accumulate met values
          const newMetTemp = newMetRolling.slice(0);
          const newMetPrec = newMetRolling.slice(1);
          const newTempAcc = hc.temp_acc.add(newMetTemp);
          const precipIncrement = newMetPrec.mul(TIME_STEP * 3600.0);
          const newPrecipAcc = hc.precip_acc.add(precipIncrement);

          // Stub GPP/vcmax with zeros (no phydro yet)
          const stubGpp = hc.sw_state.Wliq.mul(0.0);
          const stubVcmax = hc.sw_state.Wliq.mul(0.0);
          const newGppAcc = hc.gpp_acc.add(stubGpp);
          const newVcmaxAcc = hc.vcmax_acc.add(stubVcmax);
          const numGpp = hc.num_gpp.add(0.0);
          const numVcmax = hc.num_vcmax.add(0.0);

          return {
            cw_state: cwState,
            sw_state: soilResult.state,
            met_rolling: newMetRolling,
            is_first_met: notFirstMet,
            temp_acc: newTempAcc,
            precip_acc: newPrecipAcc,
            gpp_acc: newGppAcc,
            vcmax_acc: newVcmaxAcc,
            num_gpp: numGpp,
            num_vcmax: numVcmax,
            et_acc: newEtAcc,
          };
        };

        const finalHourly = lax.foriLoop(
          0, 24,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          hourlyInit as unknown as JsTree<np.Array>,
        ) as unknown as Carry;

        return [
          {
            cw_state: finalHourly.cw_state,
            sw_state: finalHourly.sw_state,
            met_rolling: finalHourly.met_rolling,
            is_first_met: finalHourly.is_first_met,
          },
          finalHourly.et_acc,
        ];
      };

      [finalCarry, ys] = lax.scan(
        dailyBody as unknown as (c: JsTree<np.Array>, x: JsTree<np.Array>) => [JsTree<np.Array>, JsTree<np.Array>],
        dailyInit as unknown as JsTree<np.Array>,
        dayXs as unknown as JsTree<np.Array>,
        { length: 1 },
      ) as unknown as [typeof dailyInit, np.Array];

      expect(ys.shape[0]).toBe(1);
      expect(finalCarry.sw_state.Wliq.shape).toEqual([]);
    } finally {
      if (ys != null) ys.dispose();
      if (finalCarry != null) tree.dispose(finalCarry);
      tree.dispose(dailyInit);
      tree.dispose(soilParams);
      tree.dispose(aeroParams);
      tree.dispose(csParams);
    }
  });

  // ----- Level 3: + pmodelHydraulicsNumerical ----- //
  it("L3: + phydro solver in hourly body", () => {
    const inputs = buildInputs();
    const { soilParams, aeroParams, csParams } = buildParams(inputs);

    using maxPoros = np.array(inputs.max_poros);
    using lai = np.array(inputs.daily_lai);
    using faparArg = lai.mul(-K_EXT);
    using faparExp = np.exp(faparArg);
    using one = np.array(1.0);
    using fapar = one.sub(faparExp);
    using hourlyTemp = np.array(inputs.hourly_temp);
    using hourlyRg = np.array(inputs.hourly_rg);
    using hourlyPrec = np.array(inputs.hourly_prec);
    using hourlyVpd = np.array(inputs.hourly_vpd);
    using hourlyPres = np.array(inputs.hourly_pres);
    using hourlyCo2 = np.array(inputs.hourly_co2);
    using hourlyWind = np.array(inputs.hourly_wind);
    using dayXs = np.array([0.0]);
    using rdark = np.array(inputs.rdark);
    using conductivity = np.array(inputs.conductivity);
    using psi50 = np.array(inputs.psi50);
    using bParam = np.array(inputs.b_param);
    using alphaCost = np.array(inputs.alpha_cost);
    using gammaCost = np.array(inputs.gamma_cost);

    const KPHIO = 0.087182;
    const C_MOLMASS = 12.0107;
    const H2O_MOLMASS = 18.01528;

    const [cwInit, swInit] = initializationSpafhy(
      inputs.soil_depth, inputs.max_poros, inputs.fc, inputs.maxpond, soilParams,
    );
    using timeStep = lai.mul(0.0).add(TIME_STEP);
    using zeroLatflow = lai.mul(0.0);

    type Carry = {
      cw_state: CanopyWaterState;
      sw_state: SoilWaterState;
      met_rolling: np.Array;
      is_first_met: np.Array;
      temp_acc: np.Array;
      precip_acc: np.Array;
      gpp_acc: np.Array;
      vcmax_acc: np.Array;
      num_gpp: np.Array;
      num_vcmax: np.Array;
      et_acc: np.Array;
    };

    const dailyInit = {
      cw_state: cwInit,
      sw_state: swInit,
      met_rolling: np.array([0.0, 0.0]),
      is_first_met: np.array(true),
    };

    const hourlyForcing = {
      temp_k: hourlyTemp, rg: hourlyRg, prec: hourlyPrec,
      vpd: hourlyVpd, pres: hourlyPres, co2: hourlyCo2, wind: hourlyWind,
    };

    let finalCarry: typeof dailyInit | null = null;
    let ys: np.Array | null = null;

    try {
      const dailyBody = (
        carry: typeof dailyInit, _x: np.Array,
      ): [typeof dailyInit, np.Array] => {
        const zeroAcc = carry.cw_state.CanopyStorage.mul(0.0);
        const hourlyInit: Carry = {
          cw_state: carry.cw_state,
          sw_state: carry.sw_state,
          met_rolling: carry.met_rolling,
          is_first_met: carry.is_first_met,
          temp_acc: zeroAcc,
          precip_acc: carry.cw_state.CanopyStorage.mul(0.0),
          gpp_acc: carry.cw_state.CanopyStorage.mul(0.0),
          vcmax_acc: carry.cw_state.CanopyStorage.mul(0.0),
          num_gpp: carry.cw_state.CanopyStorage.mul(0.0),
          num_vcmax: carry.cw_state.CanopyStorage.mul(0.0),
          et_acc: carry.cw_state.CanopyStorage.mul(0.0),
        };

        const hourlyBody = (hourIdx: np.Array, hc: Carry): Carry => {
          const tempK = lax.dynamicSlice(hourlyForcing.temp_k, [hourIdx], [1]).reshape([]);
          const rgSlice = lax.dynamicSlice(hourlyForcing.rg, [hourIdx], [1]).reshape([]);
          const prec = lax.dynamicSlice(hourlyForcing.prec, [hourIdx], [1]).reshape([]);
          const vpd = lax.dynamicSlice(hourlyForcing.vpd, [hourIdx], [1]).reshape([]);
          const pres = lax.dynamicSlice(hourlyForcing.pres, [hourIdx], [1]).reshape([]);
          const co2 = lax.dynamicSlice(hourlyForcing.co2, [hourIdx], [1]).reshape([]);
          const wind = lax.dynamicSlice(hourlyForcing.wind, [hourIdx], [1]).reshape([]);
          const tc = tempK.sub(273.15);

          // --- phydro solver ---
          const laiSafe = np.maximum(lai, LAI_GUARD);
          const ppfdBase = rgSlice.mul(2.1);
          const ppfd = ppfdBase.div(laiSafe);
          const co2Ppm = co2.mul(1.0e6);
          const phydro = pmodelHydraulicsNumerical(
            tc, ppfd, vpd, co2Ppm, pres, fapar,
            hc.sw_state.Psi,
            rdark, conductivity, psi50, bParam, alphaCost, gammaCost,
            KPHIO, "projected_newton",
          );
          const { aj, gs, vcmax: vcmaxHr } = phydro;

          // --- GPP processing ---
          const gppInner = rdark.mul(vcmaxHr);
          const gppSum = aj.add(gppInner);
          const gppScaled = gppSum.mul(C_MOLMASS).mul(1.0e-6).mul(1.0e-3).mul(lai);
          const laiPositive = lai.greater(LAI_GUARD);
          const rgPositive = rgSlice.greater(0.0);
          const hasLight = laiPositive.mul(rgPositive);
          const gppHr = np.where(hasLight, gppScaled, 0.0);
          const ajGuarded = np.where(hasLight, aj, 0.0);
          const gsGuarded = np.where(hasLight, gs, 0.0);
          const vcmaxGuarded = np.where(hasLight, vcmaxHr, 0.0);

          // --- Transpiration ---
          const rhoW = densityH2o(tc, pres);
          const vpdOverPres = vpd.div(pres);
          const trScaled = gsGuarded.mul(1.6).mul(vpdOverPres).mul(H2O_MOLMASS);
          const trRaw = trScaled.div(rhoW).mul(lai);
          const gsIsFinite = np.isfinite(gsGuarded);
          const trValid = gsIsFinite.mul(hasLight);
          const trPhydro = np.where(trValid, trRaw, 0.0);

          // --- Canopy water + soil ---
          const rn = rgSlice.mul(0.7);
          const [cwState, cwFlux] = canopyWaterFlux(
            rn, tc, prec, vpd, wind, pres,
            fapar, lai, hc.cw_state, hc.sw_state.beta, hc.sw_state.WatSto,
            aeroParams, csParams, timeStep,
          );
          const trSpafhy = trPhydro.mul(TIME_STEP * 3600.0);
          const etFromCanopy = cwFlux.SoilEvap.add(cwFlux.CanopyEvap);
          const etHr = etFromCanopy.add(trSpafhy);
          const soilResult = soilWater(
            hc.sw_state, soilParams, maxPoros,
            cwFlux.PotInfiltration, trSpafhy, cwFlux.SoilEvap,
            zeroLatflow, timeStep,
          );

          // --- Met smoothing ---
          const metPrecMelt = cwFlux.Melt.div(TIME_STEP * 3600.0);
          const metPrec = prec.add(metPrecMelt);
          const metDaily = np.stack([tc, metPrec]);
          const smoothedTempOld = hc.met_rolling.slice(0);
          const smoothedPrecOld = hc.met_rolling.slice(1);
          const metDailyTemp = metDaily.slice(0);
          const metDailyPrec = metDaily.slice(1);
          const smoothTempNew = metDailyTemp.mul(ALPHA_SMOOTH1);
          const smoothTempOldScaled = smoothedTempOld.mul(1.0 - ALPHA_SMOOTH1);
          const smoothTemp = smoothTempNew.add(smoothTempOldScaled);
          const smoothPrecNew = metDailyPrec.mul(ALPHA_SMOOTH2);
          const smoothPrecOldScaled = smoothedPrecOld.mul(1.0 - ALPHA_SMOOTH2);
          const smoothPrec = smoothPrecNew.add(smoothPrecOldScaled);
          const smoothed = np.stack([smoothTemp, smoothPrec]);
          const newMetRolling = np.where(hc.is_first_met, metDaily, smoothed);
          const notFirstMet = np.where(hc.is_first_met, false, hc.is_first_met);

          // --- Accumulate ---
          const newMetTemp = newMetRolling.slice(0);
          const newMetPrec = newMetRolling.slice(1);
          const newTempAcc = hc.temp_acc.add(newMetTemp);
          const precipIncrement = newMetPrec.mul(TIME_STEP * 3600.0);
          const newPrecipAcc = hc.precip_acc.add(precipIncrement);

          const gppValidFinite = np.isfinite(ajGuarded);
          const gppValidNonneg = ajGuarded.greaterEqual(0.0);
          const gppValid = gppValidFinite.mul(gppValidNonneg);
          const gppToAdd = np.where(gppValid, gppHr, 0.0);
          const gppCountInc = np.where(gppValid, 1.0, 0.0);
          const numGpp = hc.num_gpp.add(gppCountInc);

          const vcmaxValidFinite = np.isfinite(vcmaxGuarded);
          const vcmaxValidPositive = vcmaxGuarded.greater(0.0);
          const vcmaxValid = vcmaxValidFinite.mul(vcmaxValidPositive);
          const vcmaxToAdd = np.where(vcmaxValid, vcmaxGuarded, 0.0);
          const vcmaxCountInc = np.where(vcmaxValid, 1.0, 0.0);
          const numVcmax = hc.num_vcmax.add(vcmaxCountInc);

          const newGppAcc = hc.gpp_acc.add(gppToAdd);
          const newVcmaxAcc = hc.vcmax_acc.add(vcmaxToAdd);
          const newEtAcc = hc.et_acc.add(etHr);

          return {
            cw_state: cwState,
            sw_state: soilResult.state,
            met_rolling: newMetRolling,
            is_first_met: notFirstMet,
            temp_acc: newTempAcc,
            precip_acc: newPrecipAcc,
            gpp_acc: newGppAcc,
            vcmax_acc: newVcmaxAcc,
            num_gpp: numGpp,
            num_vcmax: numVcmax,
            et_acc: newEtAcc,
          };
        };

        const finalHourly = lax.foriLoop(
          0, 24,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          hourlyInit as unknown as JsTree<np.Array>,
        ) as unknown as Carry;

        return [
          {
            cw_state: finalHourly.cw_state,
            sw_state: finalHourly.sw_state,
            met_rolling: finalHourly.met_rolling,
            is_first_met: finalHourly.is_first_met,
          },
          finalHourly.et_acc,
        ];
      };

      [finalCarry, ys] = lax.scan(
        dailyBody as unknown as (c: JsTree<np.Array>, x: JsTree<np.Array>) => [JsTree<np.Array>, JsTree<np.Array>],
        dailyInit as unknown as JsTree<np.Array>,
        dayXs as unknown as JsTree<np.Array>,
        { length: 1 },
      ) as unknown as [typeof dailyInit, np.Array];

      expect(ys.shape[0]).toBe(1);
      expect(finalCarry.sw_state.Wliq.shape).toEqual([]);
    } finally {
      if (ys != null) ys.dispose();
      if (finalCarry != null) tree.dispose(finalCarry);
      tree.dispose(dailyInit);
      tree.dispose(soilParams);
      tree.dispose(aeroParams);
      tree.dispose(csParams);
    }
  });

  // ----- Level 4: Full daily carry shape (with stub daily post-processing) ----- //
  // First, bisection tests to find which L3→L4 change triggers the error

  // L3a: same as L3 but with forcing as scan xs (not closure-captured)
  it("L3a: forcing as scan xs instead of closure-captured", () => {
    const inputs = buildInputs();
    const { soilParams, aeroParams, csParams } = buildParams(inputs);

    using maxPoros = np.array(inputs.max_poros);
    using lai = np.array(inputs.daily_lai);
    using faparArg = lai.mul(-K_EXT);
    using faparExp = np.exp(faparArg);
    using one = np.array(1.0);
    using fapar = one.sub(faparExp);
    using rdark = np.array(inputs.rdark);
    using conductivity = np.array(inputs.conductivity);
    using psi50 = np.array(inputs.psi50);
    using bParam = np.array(inputs.b_param);
    using alphaCost = np.array(inputs.alpha_cost);
    using gammaCost = np.array(inputs.gamma_cost);

    const KPHIO = 0.087182;
    const C_MOLMASS = 12.0107;
    const H2O_MOLMASS = 18.01528;

    const [cwInit, swInit] = initializationSpafhy(
      inputs.soil_depth, inputs.max_poros, inputs.fc, inputs.maxpond, soilParams,
    );
    using timeStep = lai.mul(0.0).add(TIME_STEP);
    using zeroLatflow = lai.mul(0.0);

    type Carry = {
      cw_state: CanopyWaterState;
      sw_state: SoilWaterState;
      met_rolling: np.Array;
      is_first_met: np.Array;
      temp_acc: np.Array;
      precip_acc: np.Array;
      gpp_acc: np.Array;
      vcmax_acc: np.Array;
      num_gpp: np.Array;
      num_vcmax: np.Array;
      et_acc: np.Array;
    };

    type DailyForcingScan = {
      hourly_temp: np.Array;
      hourly_rg: np.Array;
      hourly_prec: np.Array;
      hourly_vpd: np.Array;
      hourly_pres: np.Array;
      hourly_co2: np.Array;
      hourly_wind: np.Array;
    };

    const dailyInit = {
      cw_state: cwInit,
      sw_state: swInit,
      met_rolling: np.array([0.0, 0.0]),
      is_first_met: np.array(true),
    };

    const dailyForcing: DailyForcingScan = {
      hourly_temp: np.array([inputs.hourly_temp]),
      hourly_rg: np.array([inputs.hourly_rg]),
      hourly_prec: np.array([inputs.hourly_prec]),
      hourly_vpd: np.array([inputs.hourly_vpd]),
      hourly_pres: np.array([inputs.hourly_pres]),
      hourly_co2: np.array([inputs.hourly_co2]),
      hourly_wind: np.array([inputs.hourly_wind]),
    };

    let finalCarry: typeof dailyInit | null = null;
    let ys: np.Array | null = null;

    try {
      const dailyBody = (
        carry: typeof dailyInit, forcing: DailyForcingScan,
      ): [typeof dailyInit, np.Array] => {
        const zeroAcc = carry.cw_state.CanopyStorage.mul(0.0);
        const hourlyInit: Carry = {
          cw_state: carry.cw_state,
          sw_state: carry.sw_state,
          met_rolling: carry.met_rolling,
          is_first_met: carry.is_first_met,
          temp_acc: zeroAcc,
          precip_acc: carry.cw_state.CanopyStorage.mul(0.0),
          gpp_acc: carry.cw_state.CanopyStorage.mul(0.0),
          vcmax_acc: carry.cw_state.CanopyStorage.mul(0.0),
          num_gpp: carry.cw_state.CanopyStorage.mul(0.0),
          num_vcmax: carry.cw_state.CanopyStorage.mul(0.0),
          et_acc: carry.cw_state.CanopyStorage.mul(0.0),
        };

        const hourlyBody = (hourIdx: np.Array, hc: Carry): Carry => {
          // Use forcing from scan xs instead of closure-captured
          const tempK = lax.dynamicSlice(forcing.hourly_temp, [hourIdx], [1]).reshape([]);
          const rgSlice = lax.dynamicSlice(forcing.hourly_rg, [hourIdx], [1]).reshape([]);
          const prec = lax.dynamicSlice(forcing.hourly_prec, [hourIdx], [1]).reshape([]);
          const vpd = lax.dynamicSlice(forcing.hourly_vpd, [hourIdx], [1]).reshape([]);
          const pres = lax.dynamicSlice(forcing.hourly_pres, [hourIdx], [1]).reshape([]);
          const co2 = lax.dynamicSlice(forcing.hourly_co2, [hourIdx], [1]).reshape([]);
          const wind = lax.dynamicSlice(forcing.hourly_wind, [hourIdx], [1]).reshape([]);
          const tc = tempK.sub(273.15);

          // phydro
          const laiSafe = np.maximum(lai, LAI_GUARD);
          const ppfd = rgSlice.mul(2.1).div(laiSafe);
          const co2Ppm = co2.mul(1.0e6);
          const phydro = pmodelHydraulicsNumerical(
            tc, ppfd, vpd, co2Ppm, pres, fapar,
            hc.sw_state.Psi,
            rdark, conductivity, psi50, bParam, alphaCost, gammaCost,
            KPHIO, "projected_newton",
          );
          const { aj, gs, vcmax: vcmaxHr } = phydro;

          // GPP
          const gppScaled = aj.add(rdark.mul(vcmaxHr)).mul(C_MOLMASS).mul(1e-6).mul(1e-3).mul(lai);
          const laiPositive = lai.greater(LAI_GUARD);
          const rgPositive = rgSlice.greater(0.0);
          const hasLight = laiPositive.mul(rgPositive);
          const gppHr = np.where(hasLight, gppScaled, 0.0);
          const ajGuarded = np.where(hasLight, aj, 0.0);
          const gsGuarded = np.where(hasLight, gs, 0.0);
          const vcmaxGuarded = np.where(hasLight, vcmaxHr, 0.0);

          // Transpiration
          const rhoW = densityH2o(tc, pres);
          const trScaled = gsGuarded.mul(1.6).mul(vpd.div(pres)).mul(H2O_MOLMASS);
          const trRaw = trScaled.div(rhoW).mul(lai);
          const trValid = np.isfinite(gsGuarded).mul(hasLight);
          const trPhydro = np.where(trValid, trRaw, 0.0);

          // Water
          const rn = rgSlice.mul(0.7);
          const [cwState, cwFlux] = canopyWaterFlux(
            rn, tc, prec, vpd, wind, pres,
            fapar, lai, hc.cw_state, hc.sw_state.beta, hc.sw_state.WatSto,
            aeroParams, csParams, timeStep,
          );
          const trSpafhy = trPhydro.mul(TIME_STEP * 3600.0);
          const etHr = cwFlux.SoilEvap.add(cwFlux.CanopyEvap).add(trSpafhy);
          const soilResult = soilWater(
            hc.sw_state, soilParams, maxPoros,
            cwFlux.PotInfiltration, trSpafhy, cwFlux.SoilEvap,
            zeroLatflow, timeStep,
          );

          // Met smoothing
          const metPrecMelt = cwFlux.Melt.div(TIME_STEP * 3600.0);
          const metPrec = prec.add(metPrecMelt);
          const metDaily = np.stack([tc, metPrec]);
          const smoothedTempOld = hc.met_rolling.slice(0);
          const smoothedPrecOld = hc.met_rolling.slice(1);
          const smoothTemp = metDaily.slice(0).mul(ALPHA_SMOOTH1).add(smoothedTempOld.mul(1.0 - ALPHA_SMOOTH1));
          const smoothPrec = metDaily.slice(1).mul(ALPHA_SMOOTH2).add(smoothedPrecOld.mul(1.0 - ALPHA_SMOOTH2));
          const smoothed = np.stack([smoothTemp, smoothPrec]);
          const newMetRolling = np.where(hc.is_first_met, metDaily, smoothed);
          const notFirstMet = np.where(hc.is_first_met, false, hc.is_first_met);

          // Accumulators
          const newMetTemp = newMetRolling.slice(0);
          const newMetPrec = newMetRolling.slice(1);
          const newTempAcc = hc.temp_acc.add(newMetTemp);
          const newPrecipAcc = hc.precip_acc.add(newMetPrec.mul(TIME_STEP * 3600.0));

          const gppValid = np.isfinite(ajGuarded).mul(ajGuarded.greaterEqual(0.0));
          const gppToAdd = np.where(gppValid, gppHr, 0.0);
          const numGpp = hc.num_gpp.add(np.where(gppValid, 1.0, 0.0));
          const vcmaxValid = np.isfinite(vcmaxGuarded).mul(vcmaxGuarded.greater(0.0));
          const vcmaxToAdd = np.where(vcmaxValid, vcmaxGuarded, 0.0);
          const numVcmax = hc.num_vcmax.add(np.where(vcmaxValid, 1.0, 0.0));

          return {
            cw_state: cwState,
            sw_state: soilResult.state,
            met_rolling: newMetRolling,
            is_first_met: notFirstMet,
            temp_acc: newTempAcc,
            precip_acc: newPrecipAcc,
            gpp_acc: hc.gpp_acc.add(gppToAdd),
            vcmax_acc: hc.vcmax_acc.add(vcmaxToAdd),
            num_gpp: numGpp,
            num_vcmax: numVcmax,
            et_acc: hc.et_acc.add(etHr),
          };
        };

        const finalHourly = lax.foriLoop(
          0, 24,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          hourlyInit as unknown as JsTree<np.Array>,
        ) as unknown as Carry;

        return [
          {
            cw_state: finalHourly.cw_state,
            sw_state: finalHourly.sw_state,
            met_rolling: finalHourly.met_rolling,
            is_first_met: finalHourly.is_first_met,
          },
          finalHourly.et_acc,
        ];
      };

      [finalCarry, ys] = lax.scan(
        dailyBody as unknown as (c: JsTree<np.Array>, x: JsTree<np.Array>) => [JsTree<np.Array>, JsTree<np.Array>],
        dailyInit as unknown as JsTree<np.Array>,
        dailyForcing as unknown as JsTree<np.Array>,
        { length: 1 },
      ) as unknown as [typeof dailyInit, np.Array];

      expect(ys.shape[0]).toBe(1);
      expect(finalCarry.sw_state.Wliq.shape).toEqual([]);
    } finally {
      if (ys != null) ys.dispose();
      if (finalCarry != null) tree.dispose(finalCarry);
      tree.dispose(dailyInit);
      tree.dispose(dailyForcing);
      tree.dispose(soilParams);
      tree.dispose(aeroParams);
      tree.dispose(csParams);
    }
  });

  // L3b: same as L3 but with larger daily carry (carbon fields)
  it("L3b: larger daily carry with carbon fields", () => {
    const inputs = buildInputs();
    const { soilParams, aeroParams, csParams } = buildParams(inputs);

    using maxPoros = np.array(inputs.max_poros);
    using lai = np.array(inputs.daily_lai);
    using faparArg = lai.mul(-K_EXT);
    using faparExp = np.exp(faparArg);
    using one = np.array(1.0);
    using fapar = one.sub(faparExp);
    using hourlyTemp = np.array(inputs.hourly_temp);
    using hourlyRg = np.array(inputs.hourly_rg);
    using hourlyPrec = np.array(inputs.hourly_prec);
    using hourlyVpd = np.array(inputs.hourly_vpd);
    using hourlyPres = np.array(inputs.hourly_pres);
    using hourlyCo2 = np.array(inputs.hourly_co2);
    using hourlyWind = np.array(inputs.hourly_wind);
    using dayXs = np.array([0.0]);
    using rdark = np.array(inputs.rdark);
    using conductivity = np.array(inputs.conductivity);
    using psi50 = np.array(inputs.psi50);
    using bParam = np.array(inputs.b_param);
    using alphaCost = np.array(inputs.alpha_cost);
    using gammaCost = np.array(inputs.gamma_cost);

    const KPHIO = 0.087182;
    const C_MOLMASS = 12.0107;
    const H2O_MOLMASS = 18.01528;

    const [cwInit, swInit] = initializationSpafhy(
      inputs.soil_depth, inputs.max_poros, inputs.fc, inputs.maxpond, soilParams,
    );
    using timeStep = lai.mul(0.0).add(TIME_STEP);
    using zeroLatflow = lai.mul(0.0);

    type HourlyCarryFull = {
      cw_state: CanopyWaterState;
      sw_state: SoilWaterState;
      met_rolling: np.Array;
      is_first_met: np.Array;
      temp_acc: np.Array;
      precip_acc: np.Array;
      gpp_acc: np.Array;
      vcmax_acc: np.Array;
      num_gpp: np.Array;
      num_vcmax: np.Array;
      et_acc: np.Array;
    };

    // Full daily carry shape
    const dailyInit = {
      cw_state: cwInit,
      sw_state: swInit,
      met_rolling: np.array([0.0, 0.0]),
      is_first_met: np.array(true),
      cleaf: np.array(0.0),
      croot: np.array(0.0),
      cstem: np.array(0.0),
      cgrain: np.array(0.0),
      litter_cleaf: np.array(0.0),
      litter_croot: np.array(0.0),
      compost: np.array(0.0),
      soluble: np.array(0.0),
      above: np.array(0.0),
      below: np.array(0.0),
      yield_: np.array(0.0),
      grain_fill: np.array(0.0),
      lai_alloc: np.array(0.0),
      pheno: np.array(1.0),
      cstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
      nstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
      lai_prev: np.array(0.0),
    };

    const hourlyForcing = {
      temp_k: hourlyTemp, rg: hourlyRg, prec: hourlyPrec,
      vpd: hourlyVpd, pres: hourlyPres, co2: hourlyCo2, wind: hourlyWind,
    };

    let finalCarry: typeof dailyInit | null = null;
    let ys: np.Array | null = null;

    try {
      const dailyBody = (
        carry: typeof dailyInit, _x: np.Array,
      ): [typeof dailyInit, np.Array] => {
        const zeroAcc = carry.cleaf.mul(0.0);
        const hourlyInit: HourlyCarryFull = {
          cw_state: carry.cw_state,
          sw_state: carry.sw_state,
          met_rolling: carry.met_rolling,
          is_first_met: carry.is_first_met,
          temp_acc: zeroAcc,
          precip_acc: carry.croot.mul(0.0),
          gpp_acc: carry.cstem.mul(0.0),
          vcmax_acc: carry.cgrain.mul(0.0),
          num_gpp: carry.litter_cleaf.mul(0.0),
          num_vcmax: carry.litter_croot.mul(0.0),
          et_acc: carry.compost.mul(0.0),
        };

        const hourlyBody = (hourIdx: np.Array, hc: HourlyCarryFull): HourlyCarryFull => {
          const tempK = lax.dynamicSlice(hourlyForcing.temp_k, [hourIdx], [1]).reshape([]);
          const rgSlice = lax.dynamicSlice(hourlyForcing.rg, [hourIdx], [1]).reshape([]);
          const prec = lax.dynamicSlice(hourlyForcing.prec, [hourIdx], [1]).reshape([]);
          const vpd = lax.dynamicSlice(hourlyForcing.vpd, [hourIdx], [1]).reshape([]);
          const pres = lax.dynamicSlice(hourlyForcing.pres, [hourIdx], [1]).reshape([]);
          const co2 = lax.dynamicSlice(hourlyForcing.co2, [hourIdx], [1]).reshape([]);
          const wind = lax.dynamicSlice(hourlyForcing.wind, [hourIdx], [1]).reshape([]);
          using tc = tempK.sub(273.15);

          using laiSafe = np.maximum(lai, LAI_GUARD);
          using ppfd = rgSlice.mul(2.1).div(laiSafe);
          using co2Ppm = co2.mul(1.0e6);
          const phydro = pmodelHydraulicsNumerical(
            tc, ppfd, vpd, co2Ppm, pres, fapar,
            hc.sw_state.Psi,
            rdark, conductivity, psi50, bParam, alphaCost, gammaCost,
            KPHIO, "projected_newton",
          );
          using aj = phydro.aj.ref;
          using gs = phydro.gs.ref;
          using vcmaxHr = phydro.vcmax.ref;
          tree.dispose(phydro);

          using rdarkVcmax = rdark.mul(vcmaxHr);
          using ajPlusDark = aj.add(rdarkVcmax);
          using gppMass0 = ajPlusDark.mul(C_MOLMASS);
          using gppMass1 = gppMass0.mul(1e-6);
          using gppMass2 = gppMass1.mul(1e-3);
          using gppScaled = gppMass2.mul(lai);
          using laiPositive = lai.greater(LAI_GUARD);
          using rgPositive = rgSlice.greater(0.0);
          using hasLight = laiPositive.mul(rgPositive);
          const gppHr = np.where(hasLight, gppScaled, 0.0);
          const ajGuarded = np.where(hasLight, aj, 0.0);
          const gsGuarded = np.where(hasLight, gs, 0.0);
          const vcmaxGuarded = np.where(hasLight, vcmaxHr, 0.0);

          using rhoW = densityH2o(tc, pres);
          using vpdOverPres = vpd.div(pres);
          using trScaled0 = gsGuarded.mul(1.6);
          using trScaled1 = trScaled0.mul(vpdOverPres);
          using trScaled = trScaled1.mul(H2O_MOLMASS);
          using trRawBase = trScaled.div(rhoW);
          using trRaw = trRawBase.mul(lai);
          using trValid = np.isfinite(gsGuarded).mul(hasLight);
          const trPhydro = np.where(trValid, trRaw, 0.0);

          const rn = rgSlice.mul(0.7);
          const [cwState, cwFlux] = canopyWaterFlux(
            rn, tc, prec, vpd, wind, pres,
            fapar, lai, hc.cw_state, hc.sw_state.beta, hc.sw_state.WatSto,
            aeroParams, csParams, timeStep,
          );
          const trSpafhy = trPhydro.mul(TIME_STEP * 3600.0);
          const etHr = cwFlux.SoilEvap.add(cwFlux.CanopyEvap).add(trSpafhy);
          const soilResult = soilWater(
            hc.sw_state, soilParams, maxPoros,
            cwFlux.PotInfiltration, trSpafhy, cwFlux.SoilEvap,
            zeroLatflow, timeStep,
          );

          const metPrecMelt = cwFlux.Melt.div(TIME_STEP * 3600.0);
          const metPrec = prec.add(metPrecMelt);
          const metDaily = np.stack([tc, metPrec]);
          const smoothedTempOld = hc.met_rolling.slice(0);
          const smoothedPrecOld = hc.met_rolling.slice(1);
          const smoothTemp = metDaily.slice(0).mul(ALPHA_SMOOTH1).add(smoothedTempOld.mul(1.0 - ALPHA_SMOOTH1));
          const smoothPrec = metDaily.slice(1).mul(ALPHA_SMOOTH2).add(smoothedPrecOld.mul(1.0 - ALPHA_SMOOTH2));
          const smoothed = np.stack([smoothTemp, smoothPrec]);
          const newMetRolling = np.where(hc.is_first_met, metDaily, smoothed);
          const notFirstMet = np.where(hc.is_first_met, false, hc.is_first_met);

          const newMetTemp = newMetRolling.slice(0);
          const newMetPrec = newMetRolling.slice(1);
          const newTempAcc = hc.temp_acc.add(newMetTemp);
          const newPrecipAcc = hc.precip_acc.add(newMetPrec.mul(TIME_STEP * 3600.0));

          const gppValid = np.isfinite(ajGuarded).mul(ajGuarded.greaterEqual(0.0));
          const gppToAdd = np.where(gppValid, gppHr, 0.0);
          const numGpp = hc.num_gpp.add(np.where(gppValid, 1.0, 0.0));
          const vcmaxValid = np.isfinite(vcmaxGuarded).mul(vcmaxGuarded.greater(0.0));
          const vcmaxToAdd = np.where(vcmaxValid, vcmaxGuarded, 0.0);
          const numVcmax = hc.num_vcmax.add(np.where(vcmaxValid, 1.0, 0.0));

          return {
            cw_state: cwState,
            sw_state: soilResult.state,
            met_rolling: newMetRolling,
            is_first_met: notFirstMet,
            temp_acc: newTempAcc,
            precip_acc: newPrecipAcc,
            gpp_acc: hc.gpp_acc.add(gppToAdd),
            vcmax_acc: hc.vcmax_acc.add(vcmaxToAdd),
            num_gpp: numGpp,
            num_vcmax: numVcmax,
            et_acc: hc.et_acc.add(etHr),
          };
        };

        const finalHourly = lax.foriLoop(
          0, 24,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          hourlyInit as unknown as JsTree<np.Array>,
        ) as unknown as HourlyCarryFull;

        // Pass through carbon fields unchanged
        return [
          {
            cw_state: finalHourly.cw_state,
            sw_state: finalHourly.sw_state,
            met_rolling: finalHourly.met_rolling,
            is_first_met: finalHourly.is_first_met,
            cleaf: carry.cleaf,
            croot: carry.croot,
            cstem: carry.cstem,
            cgrain: carry.cgrain,
            litter_cleaf: carry.litter_cleaf,
            litter_croot: carry.litter_croot,
            compost: carry.compost,
            soluble: carry.soluble,
            above: carry.above,
            below: carry.below,
            yield_: carry.yield_,
            grain_fill: carry.grain_fill,
            lai_alloc: carry.lai_alloc,
            pheno: carry.pheno,
            cstate: carry.cstate,
            nstate: carry.nstate,
            lai_prev: carry.lai_prev,
          },
          finalHourly.et_acc,
        ];
      };

      [finalCarry, ys] = lax.scan(
        dailyBody as unknown as (c: JsTree<np.Array>, x: JsTree<np.Array>) => [JsTree<np.Array>, JsTree<np.Array>],
        dailyInit as unknown as JsTree<np.Array>,
        dayXs as unknown as JsTree<np.Array>,
        { length: 1 },
      ) as unknown as [typeof dailyInit, np.Array];

      expect(ys.shape[0]).toBe(1);
      expect(finalCarry.sw_state.Wliq.shape).toEqual([]);
    } finally {
      if (ys != null) ys.dispose();
      if (finalCarry != null) tree.dispose(finalCarry);
      tree.dispose(dailyInit);
      tree.dispose(soilParams);
      tree.dispose(aeroParams);
      tree.dispose(csParams);
    }
  });

  it("L4: full daily carry shape with no-op daily post-processing", () => {
    const inputs = buildInputs();
    const { soilParams, aeroParams, csParams } = buildParams(inputs);

    using maxPoros = np.array(inputs.max_poros);
    using lai = np.array(inputs.daily_lai);
    using faparArg = lai.mul(-K_EXT);
    using faparExp = np.exp(faparArg);
    using one = np.array(1.0);
    using fapar = one.sub(faparExp);
    using hourlyTemp = np.array(inputs.hourly_temp);
    using hourlyRg = np.array(inputs.hourly_rg);
    using hourlyPrec = np.array(inputs.hourly_prec);
    using hourlyVpd = np.array(inputs.hourly_vpd);
    using hourlyPres = np.array(inputs.hourly_pres);
    using hourlyCo2 = np.array(inputs.hourly_co2);
    using hourlyWind = np.array(inputs.hourly_wind);
    using dayXs = np.array([0.0]);
    using rdark = np.array(inputs.rdark);
    using conductivity = np.array(inputs.conductivity);
    using psi50 = np.array(inputs.psi50);
    using bParam = np.array(inputs.b_param);
    using alphaCost = np.array(inputs.alpha_cost);
    using gammaCost = np.array(inputs.gamma_cost);

    const KPHIO = 0.087182;
    const C_MOLMASS = 12.0107;
    const H2O_MOLMASS = 18.01528;

    const [cwInit, swInit] = initializationSpafhy(
      inputs.soil_depth, inputs.max_poros, inputs.fc, inputs.maxpond, soilParams,
    );
    using timeStep = lai.mul(0.0).add(TIME_STEP);
    using zeroLatflow = lai.mul(0.0);

    // Full DailyCarry shape
    type FullDailyCarry = {
      cw_state: CanopyWaterState;
      sw_state: SoilWaterState;
      met_rolling: np.Array;
      is_first_met: np.Array;
      cleaf: np.Array;
      croot: np.Array;
      cstem: np.Array;
      cgrain: np.Array;
      litter_cleaf: np.Array;
      litter_croot: np.Array;
      compost: np.Array;
      soluble: np.Array;
      above: np.Array;
      below: np.Array;
      yield_: np.Array;
      grain_fill: np.Array;
      lai_alloc: np.Array;
      pheno: np.Array;
      cstate: np.Array;
      nstate: np.Array;
      lai_prev: np.Array;
    };

    type FullDailyOutput = {
      gpp_avg: np.Array;
      nee: np.Array;
      hetero_resp: np.Array;
      auto_resp: np.Array;
      cleaf: np.Array;
      croot: np.Array;
      cstem: np.Array;
      cgrain: np.Array;
      lai_alloc: np.Array;
      litter_cleaf: np.Array;
      litter_croot: np.Array;
      soc_total: np.Array;
      wliq: np.Array;
      psi: np.Array;
      cstate: np.Array;
      et_total: np.Array;
    };

    type HourlyCarryFull = {
      cw_state: CanopyWaterState;
      sw_state: SoilWaterState;
      met_rolling: np.Array;
      is_first_met: np.Array;
      temp_acc: np.Array;
      precip_acc: np.Array;
      gpp_acc: np.Array;
      vcmax_acc: np.Array;
      num_gpp: np.Array;
      num_vcmax: np.Array;
      et_acc: np.Array;
    };

    const dailyInit: FullDailyCarry = {
      cw_state: cwInit,
      sw_state: swInit,
      met_rolling: np.array([0.0, 0.0]),
      is_first_met: np.array(true),
      cleaf: np.array(0.0),
      croot: np.array(0.0),
      cstem: np.array(0.0),
      cgrain: np.array(0.0),
      litter_cleaf: np.array(0.0),
      litter_croot: np.array(0.0),
      compost: np.array(0.0),
      soluble: np.array(0.0),
      above: np.array(0.0),
      below: np.array(0.0),
      yield_: np.array(0.0),
      grain_fill: np.array(0.0),
      lai_alloc: np.array(0.0),
      pheno: np.array(1.0),
      cstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
      nstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
      lai_prev: np.array(0.0),
    };

    // Daily forcing needs lai for the scan xs
    type DailyForcingScan = {
      hourly_temp: np.Array;
      hourly_rg: np.Array;
      hourly_prec: np.Array;
      hourly_vpd: np.Array;
      hourly_pres: np.Array;
      hourly_co2: np.Array;
      hourly_wind: np.Array;
      lai: np.Array;
    };
    const dailyForcing: DailyForcingScan = {
      hourly_temp: np.array([inputs.hourly_temp]),
      hourly_rg: np.array([inputs.hourly_rg]),
      hourly_prec: np.array([inputs.hourly_prec]),
      hourly_vpd: np.array([inputs.hourly_vpd]),
      hourly_pres: np.array([inputs.hourly_pres]),
      hourly_co2: np.array([inputs.hourly_co2]),
      hourly_wind: np.array([inputs.hourly_wind]),
      lai: np.array([inputs.daily_lai]),
    };

    let finalCarry: FullDailyCarry | null = null;
    let ys: FullDailyOutput | null = null;

    try {
      const dailyBody = (
        carry: FullDailyCarry, forcing: DailyForcingScan,
      ): [FullDailyCarry, FullDailyOutput] => {
        const curLai = forcing.lai;
        const deltaLai = curLai.sub(carry.lai_prev);
        const curFapar_arg = curLai.mul(-K_EXT);
        const curFapar_exp = np.exp(curFapar_arg);
        const curFapar = np.array(1.0).sub(curFapar_exp);

        // Seed accumulators
        const zeroAcc = carry.cleaf.mul(0.0);
        const hourlyInit: HourlyCarryFull = {
          cw_state: carry.cw_state,
          sw_state: carry.sw_state,
          met_rolling: carry.met_rolling,
          is_first_met: carry.is_first_met,
          temp_acc: zeroAcc,
          precip_acc: carry.croot.mul(0.0),
          gpp_acc: carry.cstem.mul(0.0),
          vcmax_acc: carry.cgrain.mul(0.0),
          num_gpp: carry.litter_cleaf.mul(0.0),
          num_vcmax: carry.litter_croot.mul(0.0),
          et_acc: carry.compost.mul(0.0),
        };

        const hourlyBody = (hourIdx: np.Array, hc: HourlyCarryFull): HourlyCarryFull => {
          const tempK = lax.dynamicSlice(forcing.hourly_temp, [hourIdx], [1]).reshape([]);
          const rgSlice = lax.dynamicSlice(forcing.hourly_rg, [hourIdx], [1]).reshape([]);
          const prec = lax.dynamicSlice(forcing.hourly_prec, [hourIdx], [1]).reshape([]);
          const vpd = lax.dynamicSlice(forcing.hourly_vpd, [hourIdx], [1]).reshape([]);
          const pres = lax.dynamicSlice(forcing.hourly_pres, [hourIdx], [1]).reshape([]);
          const co2 = lax.dynamicSlice(forcing.hourly_co2, [hourIdx], [1]).reshape([]);
          const wind = lax.dynamicSlice(forcing.hourly_wind, [hourIdx], [1]).reshape([]);
          using tc = tempK.sub(273.15);

          // phydro
          using laiSafe = np.maximum(curLai, LAI_GUARD);
          using ppfd = rgSlice.mul(2.1).div(laiSafe);
          using co2Ppm = co2.mul(1.0e6);
          const phydro = pmodelHydraulicsNumerical(
            tc, ppfd, vpd, co2Ppm, pres, curFapar,
            hc.sw_state.Psi,
            rdark, conductivity, psi50, bParam, alphaCost, gammaCost,
            KPHIO, "projected_newton",
          );
          using aj = phydro.aj.ref;
          using gs = phydro.gs.ref;
          using vcmaxHr = phydro.vcmax.ref;
          tree.dispose(phydro);

          // GPP
          using rdarkVcmax = rdark.mul(vcmaxHr);
          using ajPlusDark = aj.add(rdarkVcmax);
          using gppMass0 = ajPlusDark.mul(C_MOLMASS);
          using gppMass1 = gppMass0.mul(1e-6);
          using gppMass2 = gppMass1.mul(1e-3);
          using gppScaled = gppMass2.mul(curLai);
          using laiPositive = curLai.greater(LAI_GUARD);
          using rgPositive = rgSlice.greater(0.0);
          using hasLight = laiPositive.mul(rgPositive);
          const gppHr = np.where(hasLight, gppScaled, 0.0);
          const ajGuarded = np.where(hasLight, aj, 0.0);
          const gsGuarded = np.where(hasLight, gs, 0.0);
          const vcmaxGuarded = np.where(hasLight, vcmaxHr, 0.0);

          // Transpiration
          using rhoW = densityH2o(tc, pres);
          using vpdOverPres = vpd.div(pres);
          using trScaled0 = gsGuarded.mul(1.6);
          using trScaled1 = trScaled0.mul(vpdOverPres);
          using trScaled = trScaled1.mul(H2O_MOLMASS);
          using trRawBase = trScaled.div(rhoW);
          using trRaw = trRawBase.mul(curLai);
          using trValid = np.isfinite(gsGuarded).mul(hasLight);
          const trPhydro = np.where(trValid, trRaw, 0.0);

          // Water
          const rn = rgSlice.mul(0.7);
          const curTimeStep = curLai.mul(0.0).add(TIME_STEP);
          const curZeroLatflow = curLai.mul(0.0);
          const [cwState, cwFlux] = canopyWaterFlux(
            rn, tc, prec, vpd, wind, pres,
            curFapar, curLai, hc.cw_state, hc.sw_state.beta, hc.sw_state.WatSto,
            aeroParams, csParams, curTimeStep,
          );
          const trSpafhy = trPhydro.mul(TIME_STEP * 3600.0);
          const etHr = cwFlux.SoilEvap.add(cwFlux.CanopyEvap).add(trSpafhy);
          const soilResult = soilWater(
            hc.sw_state, soilParams, maxPoros,
            cwFlux.PotInfiltration, trSpafhy, cwFlux.SoilEvap,
            curZeroLatflow, curTimeStep,
          );

          // Met smoothing
          const metPrecMelt = cwFlux.Melt.div(TIME_STEP * 3600.0);
          const metPrec = prec.add(metPrecMelt);
          const metDaily = np.stack([tc, metPrec]);
          const smoothedTempOld = hc.met_rolling.slice(0);
          const smoothedPrecOld = hc.met_rolling.slice(1);
          const smoothTemp = metDaily.slice(0).mul(ALPHA_SMOOTH1).add(smoothedTempOld.mul(1.0 - ALPHA_SMOOTH1));
          const smoothPrec = metDaily.slice(1).mul(ALPHA_SMOOTH2).add(smoothedPrecOld.mul(1.0 - ALPHA_SMOOTH2));
          const smoothed = np.stack([smoothTemp, smoothPrec]);
          const newMetRolling = np.where(hc.is_first_met, metDaily, smoothed);
          const notFirstMet = np.where(hc.is_first_met, false, hc.is_first_met);

          // Accumulators
          const newMetTemp = newMetRolling.slice(0);
          const newMetPrec = newMetRolling.slice(1);
          const newTempAcc = hc.temp_acc.add(newMetTemp);
          const newPrecipAcc = hc.precip_acc.add(newMetPrec.mul(TIME_STEP * 3600.0));

          const gppValid = np.isfinite(ajGuarded).mul(ajGuarded.greaterEqual(0.0));
          const gppToAdd = np.where(gppValid, gppHr, 0.0);
          const numGpp = hc.num_gpp.add(np.where(gppValid, 1.0, 0.0));
          const vcmaxValid = np.isfinite(vcmaxGuarded).mul(vcmaxGuarded.greater(0.0));
          const vcmaxToAdd = np.where(vcmaxValid, vcmaxGuarded, 0.0);
          const numVcmax = hc.num_vcmax.add(np.where(vcmaxValid, 1.0, 0.0));

          return {
            cw_state: cwState,
            sw_state: soilResult.state,
            met_rolling: newMetRolling,
            is_first_met: notFirstMet,
            temp_acc: newTempAcc,
            precip_acc: newPrecipAcc,
            gpp_acc: hc.gpp_acc.add(gppToAdd),
            vcmax_acc: hc.vcmax_acc.add(vcmaxToAdd),
            num_gpp: numGpp,
            num_vcmax: numVcmax,
            et_acc: hc.et_acc.add(etHr),
          };
        };

        const finalHourly = lax.foriLoop(
          0, 24,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          hourlyInit as unknown as JsTree<np.Array>,
        ) as unknown as HourlyCarryFull;

        // Stub daily post-processing: just pass through unchanged carbon fields
        const gppAvg = np.where(
          finalHourly.num_gpp.greater(0.0),
          finalHourly.gpp_acc.div(finalHourly.num_gpp),
          0.0,
        );
        const stubZero = carry.cleaf.mul(0.0);

        const nextCarry: FullDailyCarry = {
          cw_state: finalHourly.cw_state,
          sw_state: finalHourly.sw_state,
          met_rolling: finalHourly.met_rolling,
          is_first_met: finalHourly.is_first_met,
          cleaf: carry.cleaf,
          croot: carry.croot,
          cstem: carry.cstem,
          cgrain: carry.cgrain,
          litter_cleaf: carry.litter_cleaf,
          litter_croot: carry.litter_croot,
          compost: carry.compost,
          soluble: carry.soluble,
          above: carry.above,
          below: carry.below,
          yield_: carry.yield_,
          grain_fill: carry.grain_fill,
          lai_alloc: carry.lai_alloc,
          pheno: carry.pheno,
          cstate: carry.cstate,
          nstate: carry.nstate,
          lai_prev: curLai,
        };

        const output: FullDailyOutput = {
          gpp_avg: gppAvg,
          nee: stubZero,
          hetero_resp: carry.cleaf.mul(0.0),
          auto_resp: carry.cleaf.mul(0.0),
          cleaf: carry.cleaf,
          croot: carry.croot,
          cstem: carry.cstem,
          cgrain: carry.cgrain,
          lai_alloc: carry.lai_alloc,
          litter_cleaf: carry.litter_cleaf,
          litter_croot: carry.litter_croot,
          soc_total: np.sum(carry.cstate),
          wliq: finalHourly.sw_state.Wliq,
          psi: finalHourly.sw_state.Psi,
          cstate: carry.cstate,
          et_total: finalHourly.et_acc,
        };

        return [nextCarry, output];
      };

      [finalCarry as unknown, ys as unknown] = lax.scan(
        dailyBody as unknown as (c: JsTree<np.Array>, x: JsTree<np.Array>) => [JsTree<np.Array>, JsTree<np.Array>],
        dailyInit as unknown as JsTree<np.Array>,
        dailyForcing as unknown as JsTree<np.Array>,
        { length: 1 },
      );

      expect(Number.isFinite((finalCarry!.sw_state.Wliq as np.Array).js() as number)).toBe(true);
    } finally {
      if (ys != null) tree.dispose(ys);
      if (finalCarry != null) tree.dispose(finalCarry);
      tree.dispose(dailyInit);
      tree.dispose(dailyForcing);
      tree.dispose(soilParams);
      tree.dispose(aeroParams);
      tree.dispose(csParams);
    }
  });
});
