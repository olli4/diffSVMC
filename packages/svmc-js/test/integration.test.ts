import { describe, expect, it } from "vitest";
import { numpy as baseNp } from "@hamk-uas/jax-js-nonconsuming";
import { getNumericDType } from "../src/precision.js";
import { runIntegration, runIntegrationScanExperimental } from "../src/integration.js";
import integrationFixture from "../../svmc-ref/fixtures/integration.json";
import qvidjaRef from "../../../website/public/qvidja-v1-reference.json";

const YASSO_PARAM = [
  0.51, 5.19, 0.13, 0.1, 0.5, 0.0, 1.0, 1.0, 0.99, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.163, 0.0, -0.0, 0.0, 0.0, 0.0,
  0.0, 0.158, -0.002, 0.17, -0.005, 0.067, -0.0, -1.44,
  -2.0, -6.9, 0.0042, 0.0015, -2.55, 1.24, 0.25,
];

const eps = baseNp.finfo(getNumericDType()).eps;
const RTOL = 0.06;
const ATOL = 1e-12;
const NEE_ATOL = 6e-9;

function buildQvidjaRunInputs(ndays: number) {
  const defaults = qvidjaRef.defaults;
  const hourly = qvidjaRef.hourly;
  const daily = qvidjaRef.daily;
  const nhours = ndays * 24;

  return {
    hourly_temp: Array.from({ length: ndays }, (_, day) => hourly.temp_hr.slice(day * 24, (day + 1) * 24)),
    hourly_rg: Array.from({ length: ndays }, (_, day) => hourly.rg_hr.slice(day * 24, (day + 1) * 24)),
    hourly_prec: Array.from({ length: ndays }, (_, day) => hourly.prec_hr.slice(day * 24, (day + 1) * 24)),
    hourly_vpd: Array.from({ length: ndays }, (_, day) => hourly.vpd_hr.slice(day * 24, (day + 1) * 24)),
    hourly_pres: Array.from({ length: ndays }, (_, day) => hourly.pres_hr.slice(day * 24, (day + 1) * 24)),
    hourly_co2: Array.from({ length: ndays }, (_, day) => hourly.co2_hr.slice(day * 24, (day + 1) * 24)),
    hourly_wind: Array.from({ length: ndays }, (_, day) => hourly.wind_hr.slice(day * 24, (day + 1) * 24)),
    daily_lai: daily.lai_day.slice(0, ndays),
    daily_manage_type: daily.manage_type.slice(0, ndays).map((value) => Number(value)),
    daily_manage_c_in: daily.manage_c_in.slice(0, ndays),
    daily_manage_c_out: daily.manage_c_out.slice(0, ndays),
    conductivity: defaults.conductivity,
    psi50: defaults.psi50,
    b_param: defaults.b,
    alpha_cost: defaults.alpha,
    gamma_cost: defaults.gamma,
    rdark: defaults.rdark,
    soil_depth: defaults.soil_depth,
    max_poros: defaults.max_poros,
    fc: defaults.fc,
    wp: defaults.wp,
    ksat: defaults.ksat,
    n_van: 1.14,
    watres: 0.0,
    alpha_van: 5.92,
    watsat: defaults.max_poros,
    maxpond: 0.0,
    wmax: 0.5,
    wmaxsnow: 4.5,
    kmelt: 2.8934e-5,
    kfreeze: 5.79e-6,
    frac_snowliq: 0.05,
    gsoil: 5.0e-3,
    hc: 0.6,
    w_leaf: 0.01,
    rw: 0.2,
    rwmin: 0.02,
    zmeas: 2.0,
    zground: 0.1,
    zo_ground: 0.01,
    cratio_resp: defaults.cratio_resp,
    cratio_leaf: defaults.cratio_leaf,
    cratio_root: defaults.cratio_root,
    cratio_biomass: defaults.cratio_biomass,
    harvest_index: defaults.harvest_index,
    turnover_cleaf: defaults.turnover_cleaf,
    turnover_croot: defaults.turnover_croot,
    sla: defaults.sla,
    q10: defaults.q10,
    invert_option: defaults.invert_option,
    pft_is_oat: 0.0,
    yasso_param: YASSO_PARAM,
    yasso_totc: defaults.yasso_totc,
    yasso_cn_input: defaults.yasso_cn_input,
    yasso_fract_root: defaults.yasso_fract_root,
    yasso_fract_legacy: 0.0,
    yasso_tempr_c: 5.4,
    yasso_precip_day: 1.87,
    yasso_tempr_ampl: 20.0,
  };
}

describe("integration replay", () => {
  it.skip("scan experimental matches eager replay for 1 day", { timeout: 120000 }, () => {
    const inputs = buildQvidjaRunInputs(1);
    const [eagerCarry, eagerOutputs] = runIntegration(inputs);
    const [scanCarry, scanOutputs] = runIntegrationScanExperimental(inputs);

    try {
      for (const key of [
        "gpp_avg",
        "nee",
        "hetero_resp",
        "auto_resp",
        "cleaf",
        "croot",
        "cstem",
        "cgrain",
        "lai_alloc",
        "litter_cleaf",
        "litter_croot",
        "soc_total",
        "wliq",
        "psi",
        "et_total",
      ] as const) {
        expect(scanOutputs[key]).toBeAllclose(eagerOutputs[key], { rtol: RTOL, atol: eps });
      }

      expect(scanOutputs.cstate).toBeAllclose(eagerOutputs.cstate, { rtol: RTOL, atol: eps });
    } finally {
      for (const carry of [eagerCarry, scanCarry]) {
        carry.cw_state.CanopyStorage.dispose();
        carry.cw_state.SWE.dispose();
        carry.cw_state.swe_i.dispose();
        carry.cw_state.swe_l.dispose();
        carry.sw_state.WatSto.dispose();
        carry.sw_state.PondSto.dispose();
        carry.sw_state.MaxWatSto.dispose();
        carry.sw_state.MaxPondSto.dispose();
        carry.sw_state.FcSto.dispose();
        carry.sw_state.Wliq.dispose();
        carry.sw_state.Psi.dispose();
        carry.sw_state.Sat.dispose();
        carry.sw_state.Kh.dispose();
        carry.sw_state.beta.dispose();
        carry.met_rolling.dispose();
        carry.is_first_met.dispose();
        carry.cleaf.dispose();
        carry.croot.dispose();
        carry.cstem.dispose();
        carry.cgrain.dispose();
        carry.litter_cleaf.dispose();
        carry.litter_croot.dispose();
        carry.compost.dispose();
        carry.soluble.dispose();
        carry.above.dispose();
        carry.below.dispose();
        carry.yield_.dispose();
        carry.grain_fill.dispose();
        carry.lai_alloc.dispose();
        carry.pheno.dispose();
        carry.cstate.dispose();
        carry.nstate.dispose();
        carry.lai_prev.dispose();
      }

      for (const outputs of [eagerOutputs, scanOutputs]) {
        outputs.gpp_avg.dispose();
        outputs.nee.dispose();
        outputs.hetero_resp.dispose();
        outputs.auto_resp.dispose();
        outputs.cleaf.dispose();
        outputs.croot.dispose();
        outputs.cstem.dispose();
        outputs.cgrain.dispose();
        outputs.lai_alloc.dispose();
        outputs.litter_cleaf.dispose();
        outputs.litter_croot.dispose();
        outputs.soc_total.dispose();
        outputs.wliq.dispose();
        outputs.psi.dispose();
        outputs.cstate.dispose();
        outputs.et_total.dispose();
      }
    }
  });

  it("allows the projected_newton alternative on a 1-day replay", { timeout: 120000 }, () => {
    const [finalCarry, dailyOutputs] = runIntegration({
      ...buildQvidjaRunInputs(1),
      phydro_optimizer: "projected_newton",
    });

    try {
      expect(Number.isFinite((dailyOutputs.gpp_avg.js() as number[])[0])).toBe(true);
      expect(Number.isFinite((dailyOutputs.et_total.js() as number[])[0])).toBe(true);
    } finally {
      finalCarry.cw_state.CanopyStorage.dispose();
      finalCarry.cw_state.SWE.dispose();
      finalCarry.cw_state.swe_i.dispose();
      finalCarry.cw_state.swe_l.dispose();
      finalCarry.sw_state.WatSto.dispose();
      finalCarry.sw_state.PondSto.dispose();
      finalCarry.sw_state.MaxWatSto.dispose();
      finalCarry.sw_state.MaxPondSto.dispose();
      finalCarry.sw_state.FcSto.dispose();
      finalCarry.sw_state.Wliq.dispose();
      finalCarry.sw_state.Psi.dispose();
      finalCarry.sw_state.Sat.dispose();
      finalCarry.sw_state.Kh.dispose();
      finalCarry.sw_state.beta.dispose();
      finalCarry.met_rolling.dispose();
      finalCarry.is_first_met.dispose();
      finalCarry.cleaf.dispose();
      finalCarry.croot.dispose();
      finalCarry.cstem.dispose();
      finalCarry.cgrain.dispose();
      finalCarry.litter_cleaf.dispose();
      finalCarry.litter_croot.dispose();
      finalCarry.compost.dispose();
      finalCarry.soluble.dispose();
      finalCarry.above.dispose();
      finalCarry.below.dispose();
      finalCarry.yield_.dispose();
      finalCarry.grain_fill.dispose();
      finalCarry.lai_alloc.dispose();
      finalCarry.pheno.dispose();
      finalCarry.cstate.dispose();
      finalCarry.nstate.dispose();
      finalCarry.lai_prev.dispose();
      dailyOutputs.gpp_avg.dispose();
      dailyOutputs.nee.dispose();
      dailyOutputs.hetero_resp.dispose();
      dailyOutputs.auto_resp.dispose();
      dailyOutputs.cleaf.dispose();
      dailyOutputs.croot.dispose();
      dailyOutputs.cstem.dispose();
      dailyOutputs.cgrain.dispose();
      dailyOutputs.lai_alloc.dispose();
      dailyOutputs.litter_cleaf.dispose();
      dailyOutputs.litter_croot.dispose();
      dailyOutputs.soc_total.dispose();
      dailyOutputs.wliq.dispose();
      dailyOutputs.psi.dispose();
      dailyOutputs.cstate.dispose();
      dailyOutputs.et_total.dispose();
    }
  });

  it("matches the 35-day Qvidja fixture within TS float32 tolerances", { timeout: 120000 }, () => {
    const fixture = integrationFixture.integration_daily;
    const [finalCarry, dailyOutputs] = runIntegration(buildQvidjaRunInputs(35));
    const scalarKeys = [
      "gpp_avg",
      "nee",
      "hetero_resp",
      "auto_resp",
      "cleaf",
      "croot",
      "cstem",
      "cgrain",
      "lai_alloc",
      "litter_cleaf",
      "litter_croot",
      "soc_total",
      "wliq",
      "psi",
    ] as const;

    try {
      expect(fixture).toHaveLength(35);

      for (let dayIdx = 0; dayIdx < fixture.length; dayIdx += 1) {
        const expected = fixture[dayIdx].output;

        for (const key of scalarKeys) {
          const actual = (dailyOutputs[key].js() as number[])[dayIdx];
          const expectedValue = expected[key];

          if (key === "nee" || key === "hetero_resp" || key === "auto_resp") {
            expect(Math.abs(actual - expectedValue)).toBeLessThan(NEE_ATOL);
          } else if (Math.abs(expectedValue) < ATOL) {
            expect(Math.abs(actual - expectedValue)).toBeLessThan(Math.max(ATOL, eps));
          } else {
            const relErr = Math.abs(actual - expectedValue) / Math.abs(expectedValue);
            expect(relErr).toBeLessThan(RTOL);
          }
        }

        expect((dailyOutputs.cstate.js() as number[][])[dayIdx]).toBeAllclose(expected.cstate, {
          rtol: RTOL,
          atol: eps,
        });
      }

      const gppSeries = dailyOutputs.gpp_avg.js() as number[];
      const neeSeries = dailyOutputs.nee.js() as number[];
      const socSeries = dailyOutputs.soc_total.js() as number[];
      const wliqSeries = dailyOutputs.wliq.js() as number[];
      const etSeries = dailyOutputs.et_total.js() as number[];

      const expectedGppSum = fixture.reduce((sum, day) => sum + day.output.gpp_avg, 0);
      const expectedNeeSum = fixture.reduce((sum, day) => sum + day.output.nee, 0);
      const actualGppSum = gppSeries.reduce((sum, value) => sum + value, 0);
      const actualNeeSum = neeSeries.reduce((sum, value) => sum + value, 0);
      const expectedFinalSoc = fixture.at(-1)?.output.soc_total ?? 0;
      const expectedFinalWliq = fixture.at(-1)?.output.wliq ?? 0;

      expect(Math.abs(actualGppSum - expectedGppSum) / Math.abs(expectedGppSum)).toBeLessThan(RTOL);
      expect(Math.abs(actualNeeSum - expectedNeeSum)).toBeLessThan(NEE_ATOL * fixture.length);
      expect(Math.abs((socSeries.at(-1) ?? 0) - expectedFinalSoc) / Math.abs(expectedFinalSoc)).toBeLessThan(RTOL);
      expect(Math.abs((wliqSeries.at(-1) ?? 0) - expectedFinalWliq) / Math.abs(expectedFinalWliq)).toBeLessThan(RTOL);
      for (const etValue of etSeries) {
        expect(Number.isFinite(etValue)).toBe(true);
        expect(etValue).toBeGreaterThanOrEqual(0.0);
      }
    } finally {
      finalCarry.cw_state.CanopyStorage.dispose();
      finalCarry.cw_state.SWE.dispose();
      finalCarry.cw_state.swe_i.dispose();
      finalCarry.cw_state.swe_l.dispose();
      finalCarry.sw_state.WatSto.dispose();
      finalCarry.sw_state.PondSto.dispose();
      finalCarry.sw_state.MaxWatSto.dispose();
      finalCarry.sw_state.MaxPondSto.dispose();
      finalCarry.sw_state.FcSto.dispose();
      finalCarry.sw_state.Wliq.dispose();
      finalCarry.sw_state.Psi.dispose();
      finalCarry.sw_state.Sat.dispose();
      finalCarry.sw_state.Kh.dispose();
      finalCarry.sw_state.beta.dispose();
      finalCarry.met_rolling.dispose();
      finalCarry.is_first_met.dispose();
      finalCarry.cleaf.dispose();
      finalCarry.croot.dispose();
      finalCarry.cstem.dispose();
      finalCarry.cgrain.dispose();
      finalCarry.litter_cleaf.dispose();
      finalCarry.litter_croot.dispose();
      finalCarry.compost.dispose();
      finalCarry.soluble.dispose();
      finalCarry.above.dispose();
      finalCarry.below.dispose();
      finalCarry.yield_.dispose();
      finalCarry.grain_fill.dispose();
      finalCarry.lai_alloc.dispose();
      finalCarry.pheno.dispose();
      finalCarry.cstate.dispose();
      finalCarry.nstate.dispose();
      finalCarry.lai_prev.dispose();
      dailyOutputs.gpp_avg.dispose();
      dailyOutputs.nee.dispose();
      dailyOutputs.hetero_resp.dispose();
      dailyOutputs.auto_resp.dispose();
      dailyOutputs.cleaf.dispose();
      dailyOutputs.croot.dispose();
      dailyOutputs.cstem.dispose();
      dailyOutputs.cgrain.dispose();
      dailyOutputs.lai_alloc.dispose();
      dailyOutputs.litter_cleaf.dispose();
      dailyOutputs.litter_croot.dispose();
      dailyOutputs.soc_total.dispose();
      dailyOutputs.wliq.dispose();
      dailyOutputs.psi.dispose();
      dailyOutputs.cstate.dispose();
      dailyOutputs.et_total.dispose();
    }
  });
});