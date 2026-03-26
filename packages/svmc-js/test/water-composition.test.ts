import { describe, it, expect } from "vitest";
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import {
  groundEvaporation,
  canopyWaterSnow,
  canopyWaterFlux,
  soilWater,
} from "../src/water/index.js";
import type {
  CanopyWaterState,
  CanopySnowParams,
  SoilWaterState,
} from "../src/water/canopy-soil.js";
import type { SoilHydroParams } from "../src/water/soil-hydraulics.js";
import type { SpafhyAeroParams } from "../src/water/aerodynamics.js";
import waterFixtures from "../../svmc-ref/fixtures/water.json";

// Composition-function tolerances (float32)
const RTOL_FLUX = 5e-3;
const RTOL_STATE = 5e-3;

describe("Water composition functions — Fortran reference", () => {
  // ── ground_evaporation ───────────────────────────────────────────

  for (const c of waterFixtures.ground_evaporation) {
    const { inputs: inp, output: exp } = c;
    it(`ground_evaporation: T=${inp.T}, SWE=${inp.SWE}`, () => {
      using tc = np.array(inp.T);
      using ae = np.array(inp.AE);
      using vpd = np.array(inp.VPD);
      using ras = np.array(inp.Ras);
      using patm = np.array(inp.P);
      using swe = np.array(inp.SWE);
      using beta = np.array(inp.beta);
      using watSto = np.array(inp.WatSto);
      using gsoil = np.array(inp.gsoil);
      using timeStep = np.array(inp.time_step);

      using result = groundEvaporation(tc, ae, vpd, ras, patm, swe, beta, watSto, gsoil, timeStep);
      expect(result).toBeAllclose(exp.SoilEvap, { rtol: RTOL_FLUX, atol: 1e-10 });
    });
  }

  // ── canopy_water_snow ────────────────────────────────────────────

  for (const c of waterFixtures.canopy_water_snow) {
    const { inputs: inp, output: exp } = c;
    it(`canopy_water_snow: T=${inp.T}, Pre=${inp.Pre}`, () => {
      const state: CanopyWaterState = {
        CanopyStorage: np.array(inp.CanopyStorage_in),
        SWE: np.array(inp.SWE_in),
        swe_i: np.array(inp.swe_i_in),
        swe_l: np.array(inp.swe_l_in),
      };
      const params: CanopySnowParams = {
        wmax: np.array(inp.wmax),
        wmaxsnow: np.array(inp.wmaxsnow),
        kmelt: np.array(inp.kmelt),
        kfreeze: np.array(inp.kfreeze),
        fracSnowliq: np.array(inp.frac_snowliq),
        gsoil: np.array(0.01), // not used in this function directly
      };
      using tc = np.array(inp.T);
      using pre = np.array(inp.Pre);
      using ae = np.array(inp.AE);
      using d = np.array(inp.D);
      using ra = np.array(inp.Ra);
      using u = np.array(inp.U);
      using lai = np.array(inp.LAI);
      using patm = np.array(inp.P);
      using timeStep = np.array(inp.time_step);

      const [newState, flux] = canopyWaterSnow(
        state, params, tc, pre, ae, d, ra, u, lai, patm, timeStep,
      );

      try {
        // State checks
        expect(newState.CanopyStorage).toBeAllclose(exp.CanopyStorage, { rtol: RTOL_STATE });
        expect(newState.SWE).toBeAllclose(exp.SWE, { rtol: RTOL_STATE, atol: 1e-10 });
        expect(newState.swe_i).toBeAllclose(exp.swe_i, { rtol: RTOL_STATE, atol: 1e-10 });
        expect(newState.swe_l).toBeAllclose(exp.swe_l, { rtol: RTOL_STATE, atol: 1e-10 });

        // Flux checks
        expect(flux.Throughfall).toBeAllclose(exp.Throughfall, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.Interception).toBeAllclose(exp.Interception, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.CanopyEvap).toBeAllclose(exp.CanopyEvap, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.Unloading).toBeAllclose(exp.Unloading, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.PotInfiltration).toBeAllclose(exp.PotInfiltration, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.Melt).toBeAllclose(exp.Melt, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.Freeze).toBeAllclose(exp.Freeze, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.mbe).toBeAllclose(exp.mbe, { rtol: RTOL_FLUX, atol: 1e-6 });
      } finally {
        // Dispose state/flux/params (always, even on assertion failure)
        newState.CanopyStorage.dispose(); newState.SWE.dispose();
        newState.swe_i.dispose(); newState.swe_l.dispose();
        flux.Throughfall.dispose(); flux.Interception.dispose();
        flux.CanopyEvap.dispose(); flux.Unloading.dispose();
        flux.PotInfiltration.dispose(); flux.Melt.dispose();
        flux.Freeze.dispose(); flux.mbe.dispose();
        state.CanopyStorage.dispose(); state.SWE.dispose();
        state.swe_i.dispose(); state.swe_l.dispose();
        params.wmax.dispose(); params.wmaxsnow.dispose();
        params.kmelt.dispose(); params.kfreeze.dispose();
        params.fracSnowliq.dispose(); params.gsoil.dispose();
      }
    });
  }

  // ── canopy_water_flux ────────────────────────────────────────────

  for (const c of waterFixtures.canopy_water_flux) {
    const { inputs: inp, output: exp } = c;
    it(`canopy_water_flux: Ta=${inp.Ta}, LAI=${inp.LAI}`, () => {
      const cwState: CanopyWaterState = {
        CanopyStorage: np.array(inp.CanopyStorage_in),
        SWE: np.array(inp.SWE_in),
        swe_i: np.array(inp.swe_i_in),
        swe_l: np.array(inp.swe_l_in),
      };
      const aeroParams: SpafhyAeroParams = {
        hc: np.array(inp.hc),
        zmeas: np.array(inp.zmeas),
        zground: np.array(inp.zground),
        zo_ground: np.array(inp.zo_ground),
        w_leaf: np.array(inp.w_leaf),
      };
      const csParams: CanopySnowParams = {
        wmax: np.array(inp.wmax),
        wmaxsnow: np.array(inp.wmaxsnow),
        kmelt: np.array(inp.kmelt),
        kfreeze: np.array(inp.kfreeze),
        fracSnowliq: np.array(inp.frac_snowliq),
        gsoil: np.array(inp.gsoil),
      };
      using rn = np.array(inp.Rn);
      using ta = np.array(inp.Ta);
      using prec = np.array(inp.Prec);
      using vpd = np.array(inp.VPD);
      using u = np.array(inp.U);
      using patm = np.array(inp.P);
      using fapar = np.array(inp.fapar);
      using lai = np.array(inp.LAI);
      using swBeta = np.array(inp.beta_in);
      using swWatSto = np.array(inp.WatSto_in);
      using timeStep = np.array(inp.time_step);

      const [newState, flux] = canopyWaterFlux(
        rn, ta, prec, vpd, u, patm, fapar, lai,
        cwState, swBeta, swWatSto, aeroParams, csParams, timeStep,
      );

      try {
        // State checks
        expect(newState.CanopyStorage).toBeAllclose(exp.CanopyStorage, { rtol: RTOL_STATE });
        expect(newState.SWE).toBeAllclose(exp.SWE, { rtol: RTOL_STATE, atol: 1e-10 });
        expect(newState.swe_i).toBeAllclose(exp.swe_i, { rtol: RTOL_STATE, atol: 1e-10 });
        expect(newState.swe_l).toBeAllclose(exp.swe_l, { rtol: RTOL_STATE, atol: 1e-10 });

        // Flux checks
        expect(flux.Throughfall).toBeAllclose(exp.Throughfall, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.Interception).toBeAllclose(exp.Interception, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.CanopyEvap).toBeAllclose(exp.CanopyEvap, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.Unloading).toBeAllclose(exp.Unloading, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.SoilEvap).toBeAllclose(exp.SoilEvap, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.PotInfiltration).toBeAllclose(exp.PotInfiltration, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.Melt).toBeAllclose(exp.Melt, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.Freeze).toBeAllclose(exp.Freeze, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(flux.mbe).toBeAllclose(exp.mbe, { rtol: RTOL_FLUX, atol: 1e-6 });
      } finally {
        // Dispose (always, even on assertion failure)
        newState.CanopyStorage.dispose(); newState.SWE.dispose();
        newState.swe_i.dispose(); newState.swe_l.dispose();
        flux.Throughfall.dispose(); flux.Interception.dispose();
        flux.CanopyEvap.dispose(); flux.Unloading.dispose();
        flux.SoilEvap.dispose(); flux.ET.dispose(); flux.Transpiration.dispose();
        flux.PotInfiltration.dispose(); flux.Melt.dispose();
        flux.Freeze.dispose(); flux.mbe.dispose();
        cwState.CanopyStorage.dispose(); cwState.SWE.dispose();
        cwState.swe_i.dispose(); cwState.swe_l.dispose();
        aeroParams.hc.dispose(); aeroParams.zmeas.dispose();
        aeroParams.zground.dispose(); aeroParams.zo_ground.dispose();
        aeroParams.w_leaf.dispose();
        csParams.wmax.dispose(); csParams.wmaxsnow.dispose();
        csParams.kmelt.dispose(); csParams.kfreeze.dispose();
        csParams.fracSnowliq.dispose(); csParams.gsoil.dispose();
      }
    });
  }

  // ── soil_water ────────────────────────────────────────────────────

  for (const c of waterFixtures.soil_water) {
    const { inputs: inp, output: exp } = c;
    it(`soil_water: potinf=${inp.potinf}, WatSto=${inp.WatSto_in}`, () => {
      const state: SoilWaterState = {
        WatSto: np.array(inp.WatSto_in),
        PondSto: np.array(inp.PondSto_in),
        MaxWatSto: np.array(inp.MaxWatSto),
        MaxPondSto: np.array(inp.MaxPondSto),
        FcSto: np.array(inp.FcSto),
        Wliq: np.array(inp.Wliq_in),
        Psi: np.array(0),
        Sat: np.array(0),
        Kh: np.array(inp.Kh_in),
        beta: np.array(0),
      };
      const soilParams: SoilHydroParams = {
        nVan: np.array(inp.n_van),
        alphaVan: np.array(inp.alpha_van),
        watsat: np.array(inp.watsat),
        watres: np.array(inp.watres),
        ksat: np.array(inp.ksat),
      };
      using maxPoros = np.array(inp.max_poros);
      using potinf = np.array(inp.potinf);
      using tr = np.array(inp.tr);
      using evap = np.array(inp.evap);
      using latflow = np.array(inp.latflow);
      using timeStep = np.array(inp.time_step);

      const result = soilWater(
        state, soilParams, maxPoros, potinf, tr, evap, latflow, timeStep,
      );

      try {
        // State checks
        expect(result.state.WatSto).toBeAllclose(exp.WatSto, { rtol: RTOL_STATE });
        expect(result.state.PondSto).toBeAllclose(exp.PondSto, { rtol: RTOL_STATE, atol: 1e-10 });
        expect(result.state.Wliq).toBeAllclose(exp.Wliq, { rtol: RTOL_STATE });
        expect(result.state.Sat).toBeAllclose(exp.Sat, { rtol: RTOL_STATE });
        expect(result.state.beta).toBeAllclose(exp.beta, { rtol: RTOL_STATE });
        expect(result.state.Psi).toBeAllclose(exp.Psi, { rtol: RTOL_STATE, atol: 1e-10 });
        expect(result.state.Kh).toBeAllclose(exp.Kh, { rtol: 0.5, atol: 1e-6 });

        // Flux checks
        expect(result.flux.Infiltration).toBeAllclose(exp.Infiltration, { rtol: RTOL_FLUX });
        expect(result.flux.Drainage).toBeAllclose(exp.Drainage, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(result.flux.ET).toBeAllclose(exp.ET, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(result.flux.Runoff).toBeAllclose(exp.Runoff, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(result.flux.LateralFlow).toBeAllclose(exp.LateralFlow, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(result.flux.mbe).toBeAllclose(exp.mbe, { rtol: RTOL_FLUX, atol: 1e-6 });

        // Inout args
        expect(result.trOut).toBeAllclose(exp.tr_out, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(result.evapOut).toBeAllclose(exp.evap_out, { rtol: RTOL_FLUX, atol: 1e-10 });
        expect(result.latflowOut).toBeAllclose(exp.latflow_out, { rtol: RTOL_FLUX, atol: 1e-10 });
      } finally {
        // Dispose (always, even on assertion failure)
        result.state.WatSto.dispose(); result.state.PondSto.dispose();
        result.state.Wliq.dispose(); result.state.Psi.dispose();
        result.state.Sat.dispose(); result.state.Kh.dispose();
        result.state.beta.dispose();
        result.flux.Infiltration.dispose(); result.flux.Runoff.dispose();
        result.flux.Drainage.dispose(); result.flux.LateralFlow.dispose();
        result.flux.ET.dispose(); result.flux.mbe.dispose();
        result.trOut.dispose(); result.evapOut.dispose(); result.latflowOut.dispose();
        state.WatSto.dispose(); state.PondSto.dispose();
        state.MaxWatSto.dispose(); state.MaxPondSto.dispose();
        state.FcSto.dispose(); state.Wliq.dispose(); state.Psi.dispose();
        state.Sat.dispose(); state.Kh.dispose(); state.beta.dispose();
        soilParams.nVan.dispose(); soilParams.alphaVan.dispose();
        soilParams.watsat.dispose(); soilParams.watres.dispose();
        soilParams.ksat.dispose();
      }
    });
  }
});
