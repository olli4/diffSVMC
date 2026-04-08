/**
 * Bisection tests between L3b (passes) and L4 (fails).
 * 
 * L3b: full carry (21 fields), closure forcing, phydro, scalar output → PASSES
 * L4:  full carry (21 fields), forcing-as-xs, phydro, structured output, curLai from xs → FAILS
 *
 * Tests here isolate each L4-specific feature on top of L3b's base:
 *  A: L4 minus structured output (scalar output, everything else from L4)  
 *  B: L4 minus curLai from xs (closure-captured lai/fapar, everything else from L4)
 *  C: L4 but with structured output only (closure forcing, no curLai)
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
const KPHIO = 0.087182;
const C_MOLMASS = 12.0107;
const H2O_MOLMASS = 18.01528;

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

type HourlyCarry = {
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

/** The full hourly body — same in all variants. */
function makeHourlyBody(
  curLai: np.Array,
  curFapar: np.Array,
  getHourlyForcing: (hourIdx: np.Array) => {
    tempK: np.Array; rg: np.Array; prec: np.Array;
    vpd: np.Array; pres: np.Array; co2: np.Array; wind: np.Array;
  },
  rdark: np.Array,
  conductivity: np.Array,
  psi50: np.Array,
  bParam: np.Array,
  alphaCost: np.Array,
  gammaCost: np.Array,
  maxPoros: np.Array,
  aeroParams: SpafhyAeroParams,
  csParams: CanopySnowParams,
  soilParams: SoilHydroParams,
  timeStep: np.Array,
  zeroLatflow: np.Array,
) {
  return (hourIdx: np.Array, hc: HourlyCarry): HourlyCarry => {
    const { tempK, rg: rgSlice, prec, vpd, pres, co2, wind } = getHourlyForcing(hourIdx);
    using tc = tempK.sub(273.15);

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

    using rhoW = densityH2o(tc, pres);
    using vpdOverPres = vpd.div(pres);
    using trScaled0 = gsGuarded.mul(1.6);
    using trScaled1 = trScaled0.mul(vpdOverPres);
    using trScaled = trScaled1.mul(H2O_MOLMASS);
    using trRawBase = trScaled.div(rhoW);
    using trRaw = trRawBase.mul(curLai);
    using trValid = np.isfinite(gsGuarded).mul(hasLight);
    const trPhydro = np.where(trValid, trRaw, 0.0);

    const rn = rgSlice.mul(0.7);
    const [cwState, cwFlux] = canopyWaterFlux(
      rn, tc, prec, vpd, wind, pres,
      curFapar, curLai, hc.cw_state, hc.sw_state.beta, hc.sw_state.WatSto,
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
}

describe("L4 bisection – isolate trigger", () => {
  /**
   * Test A: Full L4 setup but return scalar output instead of FullDailyOutput.
   * If this passes → structured output is the trigger.
   * If this fails → structured output is NOT the trigger.
   */
  it("A: L4 but scalar output", () => {
    const inputs = buildInputs();
    const { soilParams, aeroParams, csParams } = buildParams(inputs);
    using maxPoros = np.array(inputs.max_poros);
    using rdark = np.array(inputs.rdark);
    using conductivity = np.array(inputs.conductivity);
    using psi50 = np.array(inputs.psi50);
    using bParam = np.array(inputs.b_param);
    using alphaCost = np.array(inputs.alpha_cost);
    using gammaCost = np.array(inputs.gamma_cost);

    const [cwInit, swInit] = initializationSpafhy(
      inputs.soil_depth, inputs.max_poros, inputs.fc, inputs.maxpond, soilParams,
    );

    type DailyForcingScan = {
      hourly_temp: np.Array; hourly_rg: np.Array; hourly_prec: np.Array;
      hourly_vpd: np.Array; hourly_pres: np.Array; hourly_co2: np.Array;
      hourly_wind: np.Array; lai: np.Array;
    };

    const dailyInit: FullDailyCarry = {
      cw_state: cwInit, sw_state: swInit,
      met_rolling: np.array([0.0, 0.0]), is_first_met: np.array(true),
      cleaf: np.array(0.0), croot: np.array(0.0), cstem: np.array(0.0), cgrain: np.array(0.0),
      litter_cleaf: np.array(0.0), litter_croot: np.array(0.0),
      compost: np.array(0.0), soluble: np.array(0.0), above: np.array(0.0), below: np.array(0.0),
      yield_: np.array(0.0), grain_fill: np.array(0.0), lai_alloc: np.array(0.0), pheno: np.array(1.0),
      cstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]), nstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
      lai_prev: np.array(0.0),
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
    let ys: np.Array | null = null;

    try {
      const dailyBody = (
        carry: FullDailyCarry, forcing: DailyForcingScan,
      ): [FullDailyCarry, np.Array] => {
        const curLai = forcing.lai;
        const curFapar = np.array(1.0).sub(np.exp(curLai.mul(-K_EXT)));
        const curTimeStep = curLai.mul(0.0).add(TIME_STEP);
        const curZeroLatflow = curLai.mul(0.0);

        const zeroAcc = carry.cleaf.mul(0.0);
        const hourlyInit: HourlyCarry = {
          cw_state: carry.cw_state, sw_state: carry.sw_state,
          met_rolling: carry.met_rolling, is_first_met: carry.is_first_met,
          temp_acc: zeroAcc, precip_acc: carry.croot.mul(0.0),
          gpp_acc: carry.cstem.mul(0.0), vcmax_acc: carry.cgrain.mul(0.0),
          num_gpp: carry.litter_cleaf.mul(0.0), num_vcmax: carry.litter_croot.mul(0.0),
          et_acc: carry.compost.mul(0.0),
        };

        const getForcing = (hourIdx: np.Array) => ({
          tempK: lax.dynamicSlice(forcing.hourly_temp, [hourIdx], [1]).reshape([]),
          rg: lax.dynamicSlice(forcing.hourly_rg, [hourIdx], [1]).reshape([]),
          prec: lax.dynamicSlice(forcing.hourly_prec, [hourIdx], [1]).reshape([]),
          vpd: lax.dynamicSlice(forcing.hourly_vpd, [hourIdx], [1]).reshape([]),
          pres: lax.dynamicSlice(forcing.hourly_pres, [hourIdx], [1]).reshape([]),
          co2: lax.dynamicSlice(forcing.hourly_co2, [hourIdx], [1]).reshape([]),
          wind: lax.dynamicSlice(forcing.hourly_wind, [hourIdx], [1]).reshape([]),
        });

        const hourlyBody = makeHourlyBody(
          curLai, curFapar, getForcing,
          rdark, conductivity, psi50, bParam, alphaCost, gammaCost,
          maxPoros, aeroParams, csParams, soilParams,
          curTimeStep, curZeroLatflow,
        );

        const finalHourly = lax.foriLoop(
          0, 24,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          hourlyInit as unknown as JsTree<np.Array>,
        ) as unknown as HourlyCarry;

        const nextCarry: FullDailyCarry = {
          cw_state: finalHourly.cw_state, sw_state: finalHourly.sw_state,
          met_rolling: finalHourly.met_rolling, is_first_met: finalHourly.is_first_met,
          cleaf: carry.cleaf, croot: carry.croot, cstem: carry.cstem, cgrain: carry.cgrain,
          litter_cleaf: carry.litter_cleaf, litter_croot: carry.litter_croot,
          compost: carry.compost, soluble: carry.soluble, above: carry.above, below: carry.below,
          yield_: carry.yield_, grain_fill: carry.grain_fill, lai_alloc: carry.lai_alloc,
          pheno: carry.pheno, cstate: carry.cstate, nstate: carry.nstate,
          lai_prev: curLai,
        };

        // SCALAR output (not structured)
        return [nextCarry, finalHourly.et_acc];
      };

      [finalCarry as unknown, ys] = lax.scan(
        dailyBody as unknown as (c: JsTree<np.Array>, x: JsTree<np.Array>) => [JsTree<np.Array>, JsTree<np.Array>],
        dailyInit as unknown as JsTree<np.Array>,
        dailyForcing as unknown as JsTree<np.Array>,
        { length: 1 },
      );

      expect(Number.isFinite((finalCarry!.sw_state.Wliq as np.Array).js() as number)).toBe(true);
    } finally {
      if (ys != null) (ys as np.Array).dispose();
      if (finalCarry != null) tree.dispose(finalCarry);
      tree.dispose(dailyInit);
      tree.dispose(dailyForcing);
      tree.dispose(soilParams);
      tree.dispose(aeroParams);
      tree.dispose(csParams);
    }
  });

  /**
   * Test B: Full L4 setup but use closure-captured lai/fapar instead of forcing.lai.
   * If this passes → curLai from xs is the trigger.
   * If this fails → curLai from xs is NOT the trigger.
   */
  it("B: L4 but closure-captured lai/fapar", () => {
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
    using timeStep = lai.mul(0.0).add(TIME_STEP);
    using zeroLatflow = lai.mul(0.0);

    const [cwInit, swInit] = initializationSpafhy(
      inputs.soil_depth, inputs.max_poros, inputs.fc, inputs.maxpond, soilParams,
    );

    type DailyForcingScan = {
      hourly_temp: np.Array; hourly_rg: np.Array; hourly_prec: np.Array;
      hourly_vpd: np.Array; hourly_pres: np.Array; hourly_co2: np.Array;
      hourly_wind: np.Array;
      // NO lai field
    };

    type FullDailyOutput = {
      gpp_avg: np.Array; nee: np.Array; hetero_resp: np.Array; auto_resp: np.Array;
      cleaf: np.Array; croot: np.Array; cstem: np.Array; cgrain: np.Array;
      lai_alloc: np.Array; litter_cleaf: np.Array; litter_croot: np.Array;
      soc_total: np.Array; wliq: np.Array; psi: np.Array; cstate: np.Array;
      et_total: np.Array;
    };

    const dailyInit: FullDailyCarry = {
      cw_state: cwInit, sw_state: swInit,
      met_rolling: np.array([0.0, 0.0]), is_first_met: np.array(true),
      cleaf: np.array(0.0), croot: np.array(0.0), cstem: np.array(0.0), cgrain: np.array(0.0),
      litter_cleaf: np.array(0.0), litter_croot: np.array(0.0),
      compost: np.array(0.0), soluble: np.array(0.0), above: np.array(0.0), below: np.array(0.0),
      yield_: np.array(0.0), grain_fill: np.array(0.0), lai_alloc: np.array(0.0), pheno: np.array(1.0),
      cstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]), nstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
      lai_prev: np.array(0.0),
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

    let finalCarry: FullDailyCarry | null = null;
    let ys: FullDailyOutput | null = null;

    try {
      const dailyBody = (
        carry: FullDailyCarry, forcing: DailyForcingScan,
      ): [FullDailyCarry, FullDailyOutput] => {
        const zeroAcc = carry.cleaf.mul(0.0);
        const hourlyInit: HourlyCarry = {
          cw_state: carry.cw_state, sw_state: carry.sw_state,
          met_rolling: carry.met_rolling, is_first_met: carry.is_first_met,
          temp_acc: zeroAcc, precip_acc: carry.croot.mul(0.0),
          gpp_acc: carry.cstem.mul(0.0), vcmax_acc: carry.cgrain.mul(0.0),
          num_gpp: carry.litter_cleaf.mul(0.0), num_vcmax: carry.litter_croot.mul(0.0),
          et_acc: carry.compost.mul(0.0),
        };

        const getForcing = (hourIdx: np.Array) => ({
          tempK: lax.dynamicSlice(forcing.hourly_temp, [hourIdx], [1]).reshape([]),
          rg: lax.dynamicSlice(forcing.hourly_rg, [hourIdx], [1]).reshape([]),
          prec: lax.dynamicSlice(forcing.hourly_prec, [hourIdx], [1]).reshape([]),
          vpd: lax.dynamicSlice(forcing.hourly_vpd, [hourIdx], [1]).reshape([]),
          pres: lax.dynamicSlice(forcing.hourly_pres, [hourIdx], [1]).reshape([]),
          co2: lax.dynamicSlice(forcing.hourly_co2, [hourIdx], [1]).reshape([]),
          wind: lax.dynamicSlice(forcing.hourly_wind, [hourIdx], [1]).reshape([]),
        });

        // Use closure-captured lai/fapar (NOT from forcing)
        const hourlyBody = makeHourlyBody(
          lai, fapar, getForcing,
          rdark, conductivity, psi50, bParam, alphaCost, gammaCost,
          maxPoros, aeroParams, csParams, soilParams,
          timeStep, zeroLatflow,
        );

        const finalHourly = lax.foriLoop(
          0, 24,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          hourlyInit as unknown as JsTree<np.Array>,
        ) as unknown as HourlyCarry;

        const gppAvg = np.where(
          finalHourly.num_gpp.greater(0.0),
          finalHourly.gpp_acc.div(finalHourly.num_gpp),
          0.0,
        );

        const nextCarry: FullDailyCarry = {
          cw_state: finalHourly.cw_state, sw_state: finalHourly.sw_state,
          met_rolling: finalHourly.met_rolling, is_first_met: finalHourly.is_first_met,
          cleaf: carry.cleaf, croot: carry.croot, cstem: carry.cstem, cgrain: carry.cgrain,
          litter_cleaf: carry.litter_cleaf, litter_croot: carry.litter_croot,
          compost: carry.compost, soluble: carry.soluble, above: carry.above, below: carry.below,
          yield_: carry.yield_, grain_fill: carry.grain_fill, lai_alloc: carry.lai_alloc,
          pheno: carry.pheno, cstate: carry.cstate, nstate: carry.nstate,
          lai_prev: lai,
        };

        const output: FullDailyOutput = {
          gpp_avg: gppAvg,
          nee: carry.cleaf.mul(0.0),
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

  /**
   * Test C: Closure-captured lai/fapar + closure-captured hourly forcing + structured output + full carry.
   * This is L3b + structured output. Minimal addition.
   * If this passes → structured output alone is NOT the trigger.
   */
  it("C: L3b + structured output only", () => {
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
    using rdark = np.array(inputs.rdark);
    using conductivity = np.array(inputs.conductivity);
    using psi50 = np.array(inputs.psi50);
    using bParam = np.array(inputs.b_param);
    using alphaCost = np.array(inputs.alpha_cost);
    using gammaCost = np.array(inputs.gamma_cost);
    using timeStep = lai.mul(0.0).add(TIME_STEP);
    using zeroLatflow = lai.mul(0.0);

    const [cwInit, swInit] = initializationSpafhy(
      inputs.soil_depth, inputs.max_poros, inputs.fc, inputs.maxpond, soilParams,
    );

    type FullDailyOutput = {
      gpp_avg: np.Array; nee: np.Array; hetero_resp: np.Array; auto_resp: np.Array;
      cleaf: np.Array; croot: np.Array; cstem: np.Array; cgrain: np.Array;
      lai_alloc: np.Array; litter_cleaf: np.Array; litter_croot: np.Array;
      soc_total: np.Array; wliq: np.Array; psi: np.Array; cstate: np.Array;
      et_total: np.Array;
    };

    const dailyInit: FullDailyCarry = {
      cw_state: cwInit, sw_state: swInit,
      met_rolling: np.array([0.0, 0.0]), is_first_met: np.array(true),
      cleaf: np.array(0.0), croot: np.array(0.0), cstem: np.array(0.0), cgrain: np.array(0.0),
      litter_cleaf: np.array(0.0), litter_croot: np.array(0.0),
      compost: np.array(0.0), soluble: np.array(0.0), above: np.array(0.0), below: np.array(0.0),
      yield_: np.array(0.0), grain_fill: np.array(0.0), lai_alloc: np.array(0.0), pheno: np.array(1.0),
      cstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]), nstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
      lai_prev: np.array(0.0),
    };
    using dayXs = np.array([0.0]);

    let finalCarry: FullDailyCarry | null = null;
    let ys: FullDailyOutput | null = null;

    try {
      const dailyBody = (
        carry: FullDailyCarry, _x: np.Array,
      ): [FullDailyCarry, FullDailyOutput] => {
        const zeroAcc = carry.cleaf.mul(0.0);
        const hourlyInit: HourlyCarry = {
          cw_state: carry.cw_state, sw_state: carry.sw_state,
          met_rolling: carry.met_rolling, is_first_met: carry.is_first_met,
          temp_acc: zeroAcc, precip_acc: carry.croot.mul(0.0),
          gpp_acc: carry.cstem.mul(0.0), vcmax_acc: carry.cgrain.mul(0.0),
          num_gpp: carry.litter_cleaf.mul(0.0), num_vcmax: carry.litter_croot.mul(0.0),
          et_acc: carry.compost.mul(0.0),
        };

        const getForcing = (hourIdx: np.Array) => ({
          tempK: lax.dynamicSlice(hourlyTemp, [hourIdx], [1]).reshape([]),
          rg: lax.dynamicSlice(hourlyRg, [hourIdx], [1]).reshape([]),
          prec: lax.dynamicSlice(hourlyPrec, [hourIdx], [1]).reshape([]),
          vpd: lax.dynamicSlice(hourlyVpd, [hourIdx], [1]).reshape([]),
          pres: lax.dynamicSlice(hourlyPres, [hourIdx], [1]).reshape([]),
          co2: lax.dynamicSlice(hourlyCo2, [hourIdx], [1]).reshape([]),
          wind: lax.dynamicSlice(hourlyWind, [hourIdx], [1]).reshape([]),
        });

        const hourlyBody = makeHourlyBody(
          lai, fapar, getForcing,
          rdark, conductivity, psi50, bParam, alphaCost, gammaCost,
          maxPoros, aeroParams, csParams, soilParams,
          timeStep, zeroLatflow,
        );

        const finalHourly = lax.foriLoop(
          0, 24,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          hourlyInit as unknown as JsTree<np.Array>,
        ) as unknown as HourlyCarry;

        const gppAvg = np.where(
          finalHourly.num_gpp.greater(0.0),
          finalHourly.gpp_acc.div(finalHourly.num_gpp),
          0.0,
        );

        const nextCarry: FullDailyCarry = {
          cw_state: finalHourly.cw_state, sw_state: finalHourly.sw_state,
          met_rolling: finalHourly.met_rolling, is_first_met: finalHourly.is_first_met,
          cleaf: carry.cleaf, croot: carry.croot, cstem: carry.cstem, cgrain: carry.cgrain,
          litter_cleaf: carry.litter_cleaf, litter_croot: carry.litter_croot,
          compost: carry.compost, soluble: carry.soluble, above: carry.above, below: carry.below,
          yield_: carry.yield_, grain_fill: carry.grain_fill, lai_alloc: carry.lai_alloc,
          pheno: carry.pheno, cstate: carry.cstate, nstate: carry.nstate,
          lai_prev: lai,
        };

        const output: FullDailyOutput = {
          gpp_avg: gppAvg,
          nee: carry.cleaf.mul(0.0),
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
        dayXs as unknown as JsTree<np.Array>,
        { length: 1 },
      );

      expect(Number.isFinite((finalCarry!.sw_state.Wliq as np.Array).js() as number)).toBe(true);
    } finally {
      if (ys != null) tree.dispose(ys);
      if (finalCarry != null) tree.dispose(finalCarry);
      tree.dispose(dailyInit);
      tree.dispose(soilParams);
      tree.dispose(aeroParams);
      tree.dispose(csParams);
    }
  });

  /**
   * Test D: Full L4 setup but structured output with NO carry references.
   * Output only references foriLoop results, not carry fields.
   * If this passes → carry aliasing in output + curLai is the trigger.
   * If this fails → curLai + any structured output is enough.
   */
  it("D: L4 but output without carry references", () => {
    const inputs = buildInputs();
    const { soilParams, aeroParams, csParams } = buildParams(inputs);
    using maxPoros = np.array(inputs.max_poros);
    using rdark = np.array(inputs.rdark);
    using conductivity = np.array(inputs.conductivity);
    using psi50 = np.array(inputs.psi50);
    using bParam = np.array(inputs.b_param);
    using alphaCost = np.array(inputs.alpha_cost);
    using gammaCost = np.array(inputs.gamma_cost);

    const [cwInit, swInit] = initializationSpafhy(
      inputs.soil_depth, inputs.max_poros, inputs.fc, inputs.maxpond, soilParams,
    );

    type DailyForcingScan = {
      hourly_temp: np.Array; hourly_rg: np.Array; hourly_prec: np.Array;
      hourly_vpd: np.Array; hourly_pres: np.Array; hourly_co2: np.Array;
      hourly_wind: np.Array; lai: np.Array;
    };

    type SmallOutput = {
      gpp_avg: np.Array;
      et_total: np.Array;
      wliq: np.Array;
      psi: np.Array;
    };

    const dailyInit: FullDailyCarry = {
      cw_state: cwInit, sw_state: swInit,
      met_rolling: np.array([0.0, 0.0]), is_first_met: np.array(true),
      cleaf: np.array(0.0), croot: np.array(0.0), cstem: np.array(0.0), cgrain: np.array(0.0),
      litter_cleaf: np.array(0.0), litter_croot: np.array(0.0),
      compost: np.array(0.0), soluble: np.array(0.0), above: np.array(0.0), below: np.array(0.0),
      yield_: np.array(0.0), grain_fill: np.array(0.0), lai_alloc: np.array(0.0), pheno: np.array(1.0),
      cstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]), nstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
      lai_prev: np.array(0.0),
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
    let ys: SmallOutput | null = null;

    try {
      const dailyBody = (
        carry: FullDailyCarry, forcing: DailyForcingScan,
      ): [FullDailyCarry, SmallOutput] => {
        const curLai = forcing.lai;
        const curFapar = np.array(1.0).sub(np.exp(curLai.mul(-K_EXT)));
        const curTimeStep = curLai.mul(0.0).add(TIME_STEP);
        const curZeroLatflow = curLai.mul(0.0);

        const zeroAcc = carry.cleaf.mul(0.0);
        const hourlyInit: HourlyCarry = {
          cw_state: carry.cw_state, sw_state: carry.sw_state,
          met_rolling: carry.met_rolling, is_first_met: carry.is_first_met,
          temp_acc: zeroAcc, precip_acc: carry.croot.mul(0.0),
          gpp_acc: carry.cstem.mul(0.0), vcmax_acc: carry.cgrain.mul(0.0),
          num_gpp: carry.litter_cleaf.mul(0.0), num_vcmax: carry.litter_croot.mul(0.0),
          et_acc: carry.compost.mul(0.0),
        };

        const getForcing = (hourIdx: np.Array) => ({
          tempK: lax.dynamicSlice(forcing.hourly_temp, [hourIdx], [1]).reshape([]),
          rg: lax.dynamicSlice(forcing.hourly_rg, [hourIdx], [1]).reshape([]),
          prec: lax.dynamicSlice(forcing.hourly_prec, [hourIdx], [1]).reshape([]),
          vpd: lax.dynamicSlice(forcing.hourly_vpd, [hourIdx], [1]).reshape([]),
          pres: lax.dynamicSlice(forcing.hourly_pres, [hourIdx], [1]).reshape([]),
          co2: lax.dynamicSlice(forcing.hourly_co2, [hourIdx], [1]).reshape([]),
          wind: lax.dynamicSlice(forcing.hourly_wind, [hourIdx], [1]).reshape([]),
        });

        const hourlyBody = makeHourlyBody(
          curLai, curFapar, getForcing,
          rdark, conductivity, psi50, bParam, alphaCost, gammaCost,
          maxPoros, aeroParams, csParams, soilParams,
          curTimeStep, curZeroLatflow,
        );

        const finalHourly = lax.foriLoop(
          0, 24,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          hourlyInit as unknown as JsTree<np.Array>,
        ) as unknown as HourlyCarry;

        const gppAvg = np.where(
          finalHourly.num_gpp.greater(0.0),
          finalHourly.gpp_acc.div(finalHourly.num_gpp),
          0.0,
        );

        const nextCarry: FullDailyCarry = {
          cw_state: finalHourly.cw_state, sw_state: finalHourly.sw_state,
          met_rolling: finalHourly.met_rolling, is_first_met: finalHourly.is_first_met,
          cleaf: carry.cleaf, croot: carry.croot, cstem: carry.cstem, cgrain: carry.cgrain,
          litter_cleaf: carry.litter_cleaf, litter_croot: carry.litter_croot,
          compost: carry.compost, soluble: carry.soluble, above: carry.above, below: carry.below,
          yield_: carry.yield_, grain_fill: carry.grain_fill, lai_alloc: carry.lai_alloc,
          pheno: carry.pheno, cstate: carry.cstate, nstate: carry.nstate,
          lai_prev: curLai,
        };

        // Output with NO carry references — only foriLoop results
        const output: SmallOutput = {
          gpp_avg: gppAvg,
          et_total: finalHourly.et_acc,
          wliq: finalHourly.sw_state.Wliq,
          psi: finalHourly.sw_state.Psi,
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

  /**
   * Test E: Same as D but add ONE carry reference to the output.
   * If this fails, carry aliasing + curLai is the minimal trigger.
   */
  it("E: D + one carry reference in output", () => {
    const inputs = buildInputs();
    const { soilParams, aeroParams, csParams } = buildParams(inputs);
    using maxPoros = np.array(inputs.max_poros);
    using rdark = np.array(inputs.rdark);
    using conductivity = np.array(inputs.conductivity);
    using psi50 = np.array(inputs.psi50);
    using bParam = np.array(inputs.b_param);
    using alphaCost = np.array(inputs.alpha_cost);
    using gammaCost = np.array(inputs.gamma_cost);

    const [cwInit, swInit] = initializationSpafhy(
      inputs.soil_depth, inputs.max_poros, inputs.fc, inputs.maxpond, soilParams,
    );

    type DailyForcingScan = {
      hourly_temp: np.Array; hourly_rg: np.Array; hourly_prec: np.Array;
      hourly_vpd: np.Array; hourly_pres: np.Array; hourly_co2: np.Array;
      hourly_wind: np.Array; lai: np.Array;
    };

    type SmallOutput = {
      et_total: np.Array;
      cleaf: np.Array;
    };

    const dailyInit: FullDailyCarry = {
      cw_state: cwInit, sw_state: swInit,
      met_rolling: np.array([0.0, 0.0]), is_first_met: np.array(true),
      cleaf: np.array(0.0), croot: np.array(0.0), cstem: np.array(0.0), cgrain: np.array(0.0),
      litter_cleaf: np.array(0.0), litter_croot: np.array(0.0),
      compost: np.array(0.0), soluble: np.array(0.0), above: np.array(0.0), below: np.array(0.0),
      yield_: np.array(0.0), grain_fill: np.array(0.0), lai_alloc: np.array(0.0), pheno: np.array(1.0),
      cstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]), nstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
      lai_prev: np.array(0.0),
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
    let ys: SmallOutput | null = null;

    try {
      const dailyBody = (
        carry: FullDailyCarry, forcing: DailyForcingScan,
      ): [FullDailyCarry, SmallOutput] => {
        const curLai = forcing.lai;
        const curFapar = np.array(1.0).sub(np.exp(curLai.mul(-K_EXT)));
        const curTimeStep = curLai.mul(0.0).add(TIME_STEP);
        const curZeroLatflow = curLai.mul(0.0);

        const zeroAcc = carry.cleaf.mul(0.0);
        const hourlyInit: HourlyCarry = {
          cw_state: carry.cw_state, sw_state: carry.sw_state,
          met_rolling: carry.met_rolling, is_first_met: carry.is_first_met,
          temp_acc: zeroAcc, precip_acc: carry.croot.mul(0.0),
          gpp_acc: carry.cstem.mul(0.0), vcmax_acc: carry.cgrain.mul(0.0),
          num_gpp: carry.litter_cleaf.mul(0.0), num_vcmax: carry.litter_croot.mul(0.0),
          et_acc: carry.compost.mul(0.0),
        };

        const getForcing = (hourIdx: np.Array) => ({
          tempK: lax.dynamicSlice(forcing.hourly_temp, [hourIdx], [1]).reshape([]),
          rg: lax.dynamicSlice(forcing.hourly_rg, [hourIdx], [1]).reshape([]),
          prec: lax.dynamicSlice(forcing.hourly_prec, [hourIdx], [1]).reshape([]),
          vpd: lax.dynamicSlice(forcing.hourly_vpd, [hourIdx], [1]).reshape([]),
          pres: lax.dynamicSlice(forcing.hourly_pres, [hourIdx], [1]).reshape([]),
          co2: lax.dynamicSlice(forcing.hourly_co2, [hourIdx], [1]).reshape([]),
          wind: lax.dynamicSlice(forcing.hourly_wind, [hourIdx], [1]).reshape([]),
        });

        const hourlyBody = makeHourlyBody(
          curLai, curFapar, getForcing,
          rdark, conductivity, psi50, bParam, alphaCost, gammaCost,
          maxPoros, aeroParams, csParams, soilParams,
          curTimeStep, curZeroLatflow,
        );

        const finalHourly = lax.foriLoop(
          0, 24,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          hourlyInit as unknown as JsTree<np.Array>,
        ) as unknown as HourlyCarry;

        const nextCarry: FullDailyCarry = {
          cw_state: finalHourly.cw_state, sw_state: finalHourly.sw_state,
          met_rolling: finalHourly.met_rolling, is_first_met: finalHourly.is_first_met,
          cleaf: carry.cleaf, croot: carry.croot, cstem: carry.cstem, cgrain: carry.cgrain,
          litter_cleaf: carry.litter_cleaf, litter_croot: carry.litter_croot,
          compost: carry.compost, soluble: carry.soluble, above: carry.above, below: carry.below,
          yield_: carry.yield_, grain_fill: carry.grain_fill, lai_alloc: carry.lai_alloc,
          pheno: carry.pheno, cstate: carry.cstate, nstate: carry.nstate,
          lai_prev: curLai,
        };

        const output: SmallOutput = {
          et_total: finalHourly.et_acc,
          cleaf: carry.cleaf,
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

  /**
   * Test F: Like E but with FULL L4 output (all carry aliases + post-foriLoop ops).
   */
  it("F: full L4 output shape", () => {
    const inputs = buildInputs();
    const { soilParams, aeroParams, csParams } = buildParams(inputs);
    using maxPoros = np.array(inputs.max_poros);
    using rdark = np.array(inputs.rdark);
    using conductivity = np.array(inputs.conductivity);
    using psi50 = np.array(inputs.psi50);
    using bParam = np.array(inputs.b_param);
    using alphaCost = np.array(inputs.alpha_cost);
    using gammaCost = np.array(inputs.gamma_cost);

    const [cwInit, swInit] = initializationSpafhy(
      inputs.soil_depth, inputs.max_poros, inputs.fc, inputs.maxpond, soilParams,
    );

    type DailyForcingScan = {
      hourly_temp: np.Array; hourly_rg: np.Array; hourly_prec: np.Array;
      hourly_vpd: np.Array; hourly_pres: np.Array; hourly_co2: np.Array;
      hourly_wind: np.Array; lai: np.Array;
    };

    // Reduced to 8 fields (half of original 16)
    type ReducedOutput = {
      a: np.Array; b: np.Array;
    };

    const dailyInit: FullDailyCarry = {
      cw_state: cwInit, sw_state: swInit,
      met_rolling: np.array([0.0, 0.0]), is_first_met: np.array(true),
      cleaf: np.array(0.0), croot: np.array(0.0), cstem: np.array(0.0), cgrain: np.array(0.0),
      litter_cleaf: np.array(0.0), litter_croot: np.array(0.0),
      compost: np.array(0.0), soluble: np.array(0.0), above: np.array(0.0), below: np.array(0.0),
      yield_: np.array(0.0), grain_fill: np.array(0.0), lai_alloc: np.array(0.0), pheno: np.array(1.0),
      cstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]), nstate: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
      lai_prev: np.array(0.0),
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
    let ys: ReducedOutput | null = null;

    try {
      const dailyBody = (
        carry: FullDailyCarry, forcing: DailyForcingScan,
      ): [FullDailyCarry, ReducedOutput] => {
        const curLai = forcing.lai;
        const curFapar = np.array(1.0).sub(np.exp(curLai.mul(-K_EXT)));
        const curTimeStep = curLai.mul(0.0).add(TIME_STEP);
        const curZeroLatflow = curLai.mul(0.0);

        const zeroAcc = carry.cleaf.mul(0.0);
        const hourlyInit: HourlyCarry = {
          cw_state: carry.cw_state, sw_state: carry.sw_state,
          met_rolling: carry.met_rolling, is_first_met: carry.is_first_met,
          temp_acc: zeroAcc, precip_acc: carry.croot.mul(0.0),
          gpp_acc: carry.cstem.mul(0.0), vcmax_acc: carry.cgrain.mul(0.0),
          num_gpp: carry.litter_cleaf.mul(0.0), num_vcmax: carry.litter_croot.mul(0.0),
          et_acc: carry.compost.mul(0.0),
        };

        const getForcing = (hourIdx: np.Array) => ({
          tempK: lax.dynamicSlice(forcing.hourly_temp, [hourIdx], [1]).reshape([]),
          rg: lax.dynamicSlice(forcing.hourly_rg, [hourIdx], [1]).reshape([]),
          prec: lax.dynamicSlice(forcing.hourly_prec, [hourIdx], [1]).reshape([]),
          vpd: lax.dynamicSlice(forcing.hourly_vpd, [hourIdx], [1]).reshape([]),
          pres: lax.dynamicSlice(forcing.hourly_pres, [hourIdx], [1]).reshape([]),
          co2: lax.dynamicSlice(forcing.hourly_co2, [hourIdx], [1]).reshape([]),
          wind: lax.dynamicSlice(forcing.hourly_wind, [hourIdx], [1]).reshape([]),
        });

        const hourlyBody = makeHourlyBody(
          curLai, curFapar, getForcing,
          rdark, conductivity, psi50, bParam, alphaCost, gammaCost,
          maxPoros, aeroParams, csParams, soilParams,
          curTimeStep, curZeroLatflow,
        );

        const finalHourly = lax.foriLoop(
          0, 24,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          hourlyInit as unknown as JsTree<np.Array>,
        ) as unknown as HourlyCarry;

        const gppAvg = np.where(
          finalHourly.num_gpp.greater(0.0),
          finalHourly.gpp_acc.div(finalHourly.num_gpp),
          0.0,
        );

        const nextCarry: FullDailyCarry = {
          cw_state: finalHourly.cw_state, sw_state: finalHourly.sw_state,
          met_rolling: finalHourly.met_rolling, is_first_met: finalHourly.is_first_met,
          cleaf: carry.cleaf, croot: carry.croot, cstem: carry.cstem, cgrain: carry.cgrain,
          litter_cleaf: carry.litter_cleaf, litter_croot: carry.litter_croot,
          compost: carry.compost, soluble: carry.soluble, above: carry.above, below: carry.below,
          yield_: carry.yield_, grain_fill: carry.grain_fill, lai_alloc: carry.lai_alloc,
          pheno: carry.pheno, cstate: carry.cstate, nstate: carry.nstate,
          lai_prev: curLai,
        };

        const output: ReducedOutput = {
          a: carry.cleaf,
          b: carry.cleaf,
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
