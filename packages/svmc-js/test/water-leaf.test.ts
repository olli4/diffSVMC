import { describe, it, expect } from "vitest";
import { numpy as baseNp } from "@hamk-uas/jax-js-nonconsuming";
import { getNumericDType, np } from "../src/precision.js";
import {
  eSat,
  penmanMonteith,
  soilWaterRetentionCurve,
  soilHydraulicConductivity,
  exponentialSmoothMet,
  aerodynamics,
} from "../src/water/index.js";
import type { SoilHydroParams } from "../src/water/soil-hydraulics.js";
import type { SpafhyAeroParams } from "../src/water/aerodynamics.js";
import waterFixtures from "../../svmc-ref/fixtures/water.json";

// Machine epsilon for the configured numeric dtype.
const eps = baseNp.finfo(getNumericDType()).eps;

// Per-function relative tolerances derived from critical-path operation depth
// with a ≥2× safety factor.  Each comment states the critical-path length and
// the empirically observed maximum relative error (float32 vs Fortran float64).
//
// e_sat:           5-op critical path;  observed max_rel < 1 eps
// penman_monteith: 8-op critical path;  observed max_rel < 1 eps
// soil_retention:  8-op path + Van Genuchten power amplification; observed ~26 eps
// soil_conductivity: 7-op path + Mualem conditioning; atol covers float32 underflow
// exp_smooth:      3-op critical path;  well-conditioned convex combination
// aerodynamics:   11-op critical path;  observed max_rel < 1 eps
const RTOL_ESAT = 16 * eps;
const RTOL_PM = 16 * eps;
const RTOL_SOIL_RET = 64 * eps;
const RTOL_SOIL_COND = 64 * eps;
const RTOL_SMOOTH = 8 * eps;
const RTOL_AERO = 32 * eps;

describe("Water module leaf functions — Fortran reference", () => {
  for (const c of waterFixtures.e_sat) {
    it(`e_sat: tc=${c.inputs.tc}, patm=${c.inputs.patm}`, async () => {
      using T = np.array(c.inputs.tc);
      using P = np.array(c.inputs.patm);
      const { esat, s, gamma } = eSat(T, P);
      const exp = c.output;
      expect(esat).toBeAllclose(exp.esat, { rtol: RTOL_ESAT });
      expect(s).toBeAllclose(exp.s, { rtol: RTOL_ESAT });
      expect(gamma).toBeAllclose(exp.g, { rtol: RTOL_ESAT });
      esat.dispose(); s.dispose(); gamma.dispose();
    });
  }

  for (const c of waterFixtures.penman_monteith) {
    it(`penman_monteith: tc=${c.inputs.tc}, patm=${c.inputs.patm}, vpd=${c.inputs.vpd}`, async () => {
      using AE = np.array(c.inputs.AE);
      using D = np.array(c.inputs.vpd);
      using T = np.array(c.inputs.tc);
      using Gs = np.array(c.inputs.Gs);
      using Ga = np.array(c.inputs.Ga);
      using P = np.array(c.inputs.patm);
      using result = penmanMonteith(AE, D, T, Gs, Ga, P);
      expect(result).toBeAllclose(c.output as number, { rtol: RTOL_PM });
    });
  }

  for (const c of waterFixtures.soil_water_retention_curve) {
    it(`soil_retention: vol_liq=${c.inputs.vol_liq}`, async () => {
      const params: SoilHydroParams = {
        nVan: np.array(c.inputs.n_van),
        alphaVan: np.array(c.inputs.alpha_van),
        watsat: np.array(c.inputs.watsat),
        watres: np.array(c.inputs.watres),
        ksat: np.array(1e-5),
      };
      using volLiq = np.array(c.inputs.vol_liq);
      using result = soilWaterRetentionCurve(volLiq, params);
      expect(result).toBeAllclose(c.output as number, { rtol: RTOL_SOIL_RET });
      params.nVan.dispose(); params.alphaVan.dispose();
      params.watsat.dispose(); params.watres.dispose(); params.ksat.dispose();
    });
  }

  for (const c of waterFixtures.soil_hydraulic_conductivity) {
    it(`soil_conductivity: vol_liq=${c.inputs.vol_liq}`, async () => {
      const params: SoilHydroParams = {
        nVan: np.array(c.inputs.n_van),
        alphaVan: np.array(c.inputs.alpha_van),
        watsat: np.array(c.inputs.watsat),
        watres: np.array(c.inputs.watres),
        ksat: np.array(c.inputs.ksat),
      };
      using volLiq = np.array(c.inputs.vol_liq);
      using result = soilHydraulicConductivity(volLiq, params);
      // Mualem formula is ill-conditioned near saturation boundaries: catastrophic
      // cancellation in 1-(1-S^(1/m))^m produces large condition numbers for small
      // outputs.  atol = eps provides a float32 noise floor; rtol covers the
      // well-conditioned regime.
      expect(result).toBeAllclose(c.output as number, { rtol: RTOL_SOIL_COND, atol: eps });
      params.nVan.dispose(); params.alphaVan.dispose();
      params.watsat.dispose(); params.watres.dispose(); params.ksat.dispose();
    });
  }

  for (const c of waterFixtures.exponential_smooth_met) {
    it(`exponential_smooth_met: met_ind=${c.inputs.met_ind_in}`, async () => {
      using daily = np.array(c.inputs.met_daily);
      using rolling = np.array(c.inputs.met_rolling_in);
      const { metRolling, metInd } = exponentialSmoothMet(daily, rolling, c.inputs.met_ind_in);
      const exp = c.output;
      expect(metRolling).toBeAllclose(exp.met_rolling, { rtol: RTOL_SMOOTH });
      expect(metInd).toBe(exp.met_ind);
      metRolling.dispose();
    });
  }

  for (const c of waterFixtures.aerodynamics) {
    it(`aerodynamics: LAI=${c.inputs.LAI}, Uo=${c.inputs.Uo}, hc=${c.inputs.hc}`, async () => {
      const params: SpafhyAeroParams = {
        hc: np.array(c.inputs.hc),
        zmeas: np.array(c.inputs.zmeas),
        zground: np.array(c.inputs.zground),
        zo_ground: np.array(c.inputs.zo_ground),
        w_leaf: np.array(c.inputs.w_leaf),
      };
      using LAI = np.array(c.inputs.LAI);
      using Uo = np.array(c.inputs.Uo);
      const result = aerodynamics(LAI, Uo, params);
      const exp = c.output;
      expect(result.ra).toBeAllclose(exp.ra, { rtol: RTOL_AERO });
      expect(result.rb).toBeAllclose(exp.rb, { rtol: RTOL_AERO, atol: eps });
      expect(result.ras).toBeAllclose(exp.ras, { rtol: RTOL_AERO });
      expect(result.ustar).toBeAllclose(exp.ustar, { rtol: RTOL_AERO });
      expect(result.Uh).toBeAllclose(exp.Uh, { rtol: RTOL_AERO });
      expect(result.Ug).toBeAllclose(exp.Ug, { rtol: RTOL_AERO });
      result.ra.dispose(); result.rb.dispose(); result.ras.dispose();
      result.ustar.dispose(); result.Uh.dispose(); result.Ug.dispose();
      params.hc.dispose(); params.zmeas.dispose(); params.zground.dispose();
      params.zo_ground.dispose(); params.w_leaf.dispose();
    });
  }
});
