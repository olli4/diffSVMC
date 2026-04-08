/// <reference path="./vitest.d.ts" />

import { clearCaches, lax } from "@hamk-uas/jax-js-nonconsuming";
import { describe, it, expect } from "vitest";
import { np } from "../src/precision.js";
import {
  ftempArrh,
  gammastar,
  ftempKphio,
  densityH2o,
  viscosityH2o,
  calcKmm,
  scaleConduct,
  calcGs,
  calcAssimLightLimited,
  fnProfit,
  optimiseMidtermMulti,
  quadratic,
} from "../src/phydro/index.js";
import type {
  ParEnv,
  ParPlant,
  ParCost,
  ParPhotosynth,
} from "../src/phydro/scale-conductivity.js";
import phydroFixtures from "../../svmc-ref/fixtures/phydro.json";

const KPHIO = 0.087182;

function makeParEnv(tc: number, patm: number, vpd = 1000): ParEnv {
  const tcArr = np.array(tc);
  const patmArr = np.array(patm);
  return {
    viscosityWater: viscosityH2o(tcArr, patmArr),
    densityWater: densityH2o(tcArr, patmArr),
    patm: patmArr,
    tc: tcArr,
    vpd: np.array(vpd),
  };
}

describe("P-Hydro leaf functions — Fortran reference", () => {
  for (const c of phydroFixtures.ftemp_arrh) {
    it(`ftemp_arrh: tk=${c.inputs.tk}, dha=${c.inputs.dha}`, async () => {
      using tk = np.array(c.inputs.tk);
      using dha = np.array(c.inputs.dha);
      using result = ftempArrh(tk, dha);
      expect(result).toBeAllclose(c.output as number, { rtol: 5e-3 });
    });
  }

  for (const c of phydroFixtures.gammastar) {
    it(`gammastar: tc=${c.inputs.tc}, patm=${c.inputs.patm}`, async () => {
      using tc = np.array(c.inputs.tc);
      using patm = np.array(c.inputs.patm);
      using result = gammastar(tc, patm);
      expect(result).toBeAllclose(c.output as number, { rtol: 5e-3 });
    });
  }

  for (const c of phydroFixtures.ftemp_kphio_c3) {
    it(`ftemp_kphio_c3: tc=${c.inputs.tc}`, async () => {
      using tc = np.array(c.inputs.tc);
      using result = ftempKphio(tc, false);
      expect(result).toBeAllclose(c.output as number, { rtol: 5e-5, atol: 1e-7 });
    });
  }

  for (const c of phydroFixtures.ftemp_kphio_c4) {
    it(`ftemp_kphio_c4: tc=${c.inputs.tc}`, async () => {
      using tc = np.array(c.inputs.tc);
      using result = ftempKphio(tc, true);
      expect(result).toBeAllclose(c.output as number, { rtol: 5e-5, atol: 1e-7 });
    });
  }

  for (const c of phydroFixtures.density_h2o) {
    it(`density_h2o: tc=${c.inputs.tc}, patm=${c.inputs.patm}`, async () => {
      using tc = np.array(c.inputs.tc);
      using patm = np.array(c.inputs.patm);
      using result = densityH2o(tc, patm);
      expect(result).toBeAllclose(c.output as number, { rtol: 5e-3 });
    });
  }

  for (const c of phydroFixtures.viscosity_h2o) {
    it(`viscosity_h2o: tc=${c.inputs.tc}, patm=${c.inputs.patm}`, async () => {
      using tc = np.array(c.inputs.tc);
      using patm = np.array(c.inputs.patm);
      using result = viscosityH2o(tc, patm);
      expect(result).toBeAllclose(c.output as number, { rtol: 5e-3 });
    });
  }

  for (const c of phydroFixtures.calc_kmm) {
    it(`calc_kmm: tc=${c.inputs.tc}, patm=${c.inputs.patm}`, async () => {
      using tc = np.array(c.inputs.tc);
      using patm = np.array(c.inputs.patm);
      using result = calcKmm(tc, patm);
      expect(result).toBeAllclose(c.output as number, { rtol: 5e-3 });
    });
  }

  for (const c of phydroFixtures.scale_conductivity) {
    it(`scale_conductivity: tc=${c.inputs.tc}, patm=${c.inputs.patm}`, async () => {
      const parEnv = makeParEnv(c.inputs.tc, c.inputs.patm);
      using K = np.array(c.inputs.K);
      using result = scaleConduct(K, parEnv);
      expect(result).toBeAllclose(c.output as number, { rtol: 5e-2 });
      parEnv.viscosityWater.dispose();
      parEnv.densityWater.dispose();
      parEnv.patm.dispose();
      parEnv.tc.dispose();
      parEnv.vpd.dispose();
    });
  }

  for (const c of phydroFixtures.calc_gs) {
    it(`calc_gs: tc=${c.inputs.tc}, patm=${c.inputs.patm}`, async () => {
      const parEnv = makeParEnv(c.inputs.tc, c.inputs.patm, c.inputs.vpd);
      const parPlant: ParPlant = {
        conductivity: np.array(c.inputs.conductivity),
        psi50: np.array(c.inputs.psi50),
        b: np.array(c.inputs.b),
      };
      using dpsi = np.array(c.inputs.dpsi);
      using psiSoil = np.array(c.inputs.psi_soil);
      using result = calcGs(
        dpsi, psiSoil,
        parPlant, parEnv,
      );
      expect(result).toBeAllclose(c.output as number, { rtol: 5e-2 });
      parEnv.viscosityWater.dispose(); parEnv.densityWater.dispose();
      parEnv.patm.dispose(); parEnv.tc.dispose(); parEnv.vpd.dispose();
      parPlant.conductivity.dispose(); parPlant.psi50.dispose(); parPlant.b.dispose();
    });
  }

  for (const c of phydroFixtures.calc_assim_light_limited) {
    it(`calc_assim: tc=${c.inputs.tc}, patm=${c.inputs.patm}`, async () => {
      const { tc, patm } = c.inputs;
      const parEnv = makeParEnv(tc, patm);
      const parPlant: ParPlant = {
        conductivity: np.array(4e-16), psi50: np.array(-3.46), b: np.array(2.0),
      };
      using dpsi0 = np.array(1.0);
      using psiSoil0 = np.array(-0.5);
      using gs = calcGs(dpsi0, psiSoil0, parPlant, parEnv);
      using tcArr = np.array(tc);
      using patmArr = np.array(patm);
      using kmmVal = calcKmm(tcArr, patmArr);
      using gsVal = gammastar(tcArr, patmArr);
      using kphioVal = ftempKphio(tcArr, false);
      using phi0 = kphioVal.mul(KPHIO);
      const parPs: ParPhotosynth = {
        kmm: kmmVal, gammastar: gsVal, phi0,
        Iabs: np.array(c.inputs.Iabs),
        ca: np.array(c.inputs.ca_ppm * patm * 1e-6),
        patm: np.array(patm), delta: np.array(c.inputs.delta),
      };
      using jmaxArr = np.array(c.inputs.jmax);
      const { ci, aj } = calcAssimLightLimited(gs, jmaxArr, parPs);
      const exp = c.output as { ci: number; aj: number };
      expect(ci).toBeAllclose(exp.ci, { rtol: 5e-2, atol: 1e-4 });
      expect(aj).toBeAllclose(exp.aj, { rtol: 5e-2, atol: 1e-4 });
      ci.dispose(); aj.dispose();
      parEnv.viscosityWater.dispose(); parEnv.densityWater.dispose();
      parEnv.patm.dispose(); parEnv.tc.dispose(); parEnv.vpd.dispose();
      parPlant.conductivity.dispose(); parPlant.psi50.dispose(); parPlant.b.dispose();
      parPs.Iabs.dispose(); parPs.ca.dispose(); parPs.patm.dispose(); parPs.delta.dispose();
    });
  }

  for (const c of phydroFixtures.fn_profit) {
    it(`fn_profit: tc=${c.inputs.tc}, patm=${c.inputs.patm}, hypothesis=${c.inputs.hypothesis ?? "PM"}, do_optim=${c.inputs.do_optim ?? false}`, async () => {
      const { tc, patm } = c.inputs;
      const hypothesis = (c.inputs.hypothesis ?? "PM") as "PM" | "LC";
      const parEnv = makeParEnv(tc, patm, c.inputs.vpd);
      const parPlant: ParPlant = {
        conductivity: np.array(c.inputs.conductivity),
        psi50: np.array(c.inputs.psi50),
        b: np.array(c.inputs.b),
      };
      const parCost: ParCost = { alpha: np.array(c.inputs.alpha), gamma: np.array(c.inputs.gamma) };
      using tcArr = np.array(tc);
      using patmArr = np.array(patm);
      using kmmVal = calcKmm(tcArr, patmArr);
      using gsVal = gammastar(tcArr, patmArr);
      using kphioVal = ftempKphio(tcArr, false);
      using phi0 = kphioVal.mul(KPHIO);
      const parPs: ParPhotosynth = {
        kmm: kmmVal, gammastar: gsVal, phi0,
        Iabs: np.array(c.inputs.Iabs),
        ca: np.array(c.inputs.ca_ppm * patm * 1e-6),
        patm: np.array(patm), delta: np.array(c.inputs.delta),
      };
      using logjmax = np.array(c.inputs.logjmax);
      using dpsi = np.array(c.inputs.dpsi);
      using psiSoil = np.array(c.inputs.psi_soil);
      using result = fnProfit(
        logjmax, dpsi,
        psiSoil, parCost, parPs, parPlant, parEnv,
        hypothesis,
        c.inputs.do_optim ?? false,
      );
      expect(result).toBeAllclose(c.output as number, { rtol: 5e-2, atol: 1e-4 });
      parEnv.viscosityWater.dispose(); parEnv.densityWater.dispose();
      parEnv.patm.dispose(); parEnv.tc.dispose(); parEnv.vpd.dispose();
      parPlant.conductivity.dispose(); parPlant.psi50.dispose(); parPlant.b.dispose();
      parCost.alpha.dispose(); parCost.gamma.dispose();
      parPs.Iabs.dispose(); parPs.ca.dispose(); parPs.patm.dispose(); parPs.delta.dispose();
    });
  }

  for (const c of phydroFixtures.quadratic) {
    it(`quadratic: a=${c.inputs.a}, b=${c.inputs.b}, c=${c.inputs.c}`, async () => {
      using a = np.array(c.inputs.a);
      using b = np.array(c.inputs.b);
      using cCoeff = np.array(c.inputs.c);
      using result = quadratic(a, b, cCoeff);
      expect(result).toBeAllclose(c.output as number, { rtol: 5e-3 });
    });
  }
});

// ── pmodel_hydraulics_numerical (full solver) ────────────────────────

import { pmodelHydraulicsNumerical } from "../src/phydro/index.js";

describe("P-Hydro solver — Fortran reference", () => {
  // Type assertion: fixture may not have this key yet in older snapshots
  const solverFixtures =
    (phydroFixtures as Record<string, unknown[]>)[
      "pmodel_hydraulics_numerical"
    ] ?? [];

  for (const c of solverFixtures as Array<{
    inputs: Record<string, number>;
    output: Record<string, number>;
  }>) {
    it(`pmodel tc=${c.inputs.tc} vpd=${c.inputs.vpd} psi=${c.inputs.psi_soil}`, async () => {
      const result = pmodelHydraulicsNumerical(
        c.inputs.tc,
        c.inputs.ppfd,
        c.inputs.vpd,
        c.inputs.co2,
        c.inputs.sp,
        c.inputs.fapar,
        c.inputs.psi_soil,
        c.inputs.rdark_leaf,
      );
      // Default TS mode uses float32 + FD gradients; Fortran uses float64 + FD gradients.
      // Tolerance reflects both optimizer convergence diff and default float32 precision.
      const rtol = 5e-2;
      const atol = 1e-4;
      for (const key of [
        "jmax",
        "dpsi",
        "gs",
        "aj",
        "ci",
        "chi",
        "vcmax",
        "profit",
      ] as const) {
        const tsKey =
          key === "chi_jmax_lim" ? "chiJmaxLim" : key;
        const got = (result as Record<string, np.Array>)[tsKey].item() as number;
        const expected = c.output[key];
        const err =
          Math.abs(got - expected) /
          Math.max(Math.abs(expected), atol);
        if (err > rtol) {
          throw new Error(
            `${key}: TS=${got} vs Fortran=${expected} (relErr=${err.toExponential(2)})`,
          );
        }
      }
      // Dispose all returned arrays
      result.jmax.dispose();
      result.dpsi.dispose();
      result.gs.dispose();
      result.aj.dispose();
      result.ci.dispose();
      result.chi.dispose();
      result.vcmax.dispose();
      result.profit.dispose();
      result.chiJmaxLim.dispose();
    });
  }

  it("profit should be positive at benign conditions", async () => {
    const result = pmodelHydraulicsNumerical(
      20.0,
      300.0,
      1000.0,
      400.0,
      101325.0,
      0.9,
      -0.5,
      0.015,
    );
    const profit = result.profit.item() as number;
    expect(profit).toBeGreaterThan(0);
    const jmax = result.jmax.item() as number;
    expect(jmax).toBeGreaterThan(0);
    result.jmax.dispose();
    result.dpsi.dispose();
    result.gs.dispose();
    result.aj.dispose();
    result.ci.dispose();
    result.chi.dispose();
    result.vcmax.dispose();
    result.profit.dispose();
    result.chiJmaxLim.dispose();
  });

  it("projected_newton remains selectable", async () => {
    const result = pmodelHydraulicsNumerical(
      20.0,
      300.0,
      1000.0,
      400.0,
      101325.0,
      0.9,
      -0.5,
      0.015,
      4e-16,
      -3.46,
      2.0,
      0.1,
      0.5,
      0.087182,
      "projected_newton",
    );

    for (const key of ["jmax", "dpsi", "gs", "aj", "ci", "chi", "vcmax", "profit"] as const) {
      expect(Number.isFinite((result[key] as np.Array).item() as number)).toBe(true);
    }

    result.jmax.dispose();
    result.dpsi.dispose();
    result.gs.dispose();
    result.aj.dispose();
    result.ci.dispose();
    result.chi.dispose();
    result.vcmax.dispose();
    result.profit.dispose();
    result.chiJmaxLim.dispose();
  });

  // Regression guard: v0.12.10 fixed the nested scan + solver leak path that
  // previously forced this diagnostic to stay skipped.
  it("optimiseMidtermMulti works inside lax.scan when params are arrays", () => {
    using tcArr = np.array(20.0);
    using patmArr = np.array(101325.0);
    using psiSoil = np.array(-0.5);
    using vpdArr = np.array(1000.0);
    using ppfdArr = np.array(300.0);
    using faparArr = np.array(0.9);
    using co2Arr = np.array(400.0);
    using rdarkArr = np.array(0.015);
    using conductivityArr = np.array(4e-16);
    using psi50Arr = np.array(-3.46);
    using bArr = np.array(2.0);
    using alphaArr = np.array(0.1);
    using gammaArr = np.array(0.5);
    using xs = np.array([0.0]);
    using init = np.array(0.0);

    const parCost: ParCost = { alpha: alphaArr, gamma: gammaArr };
    const parPlant: ParPlant = {
      conductivity: conductivityArr,
      psi50: psi50Arr,
      b: bArr,
    };
    const parEnv: ParEnv = {
      viscosityWater: viscosityH2o(tcArr, patmArr),
      densityWater: densityH2o(tcArr, patmArr),
      patm: patmArr,
      tc: tcArr,
      vpd: vpdArr,
    };
    const parPhotosynth: ParPhotosynth = {
      kmm: calcKmm(tcArr, patmArr),
      gammastar: gammastar(tcArr, patmArr),
      phi0: ftempKphio(tcArr, false).mul(KPHIO),
      Iabs: ppfdArr.mul(faparArr),
      ca: co2Arr.mul(patmArr.mul(1e-6)),
      patm: patmArr,
      delta: rdarkArr,
    };

    try {
      const [carry, ys] = lax.scan((c: np.Array, _x: np.Array): [np.Array, np.Array] => {
        const result = optimiseMidtermMulti(psiSoil, parCost, parPhotosynth, parPlant, parEnv);
        return [c, result.jmax];
      }, init, xs, { length: 1 });

      carry.dispose();
      using outputs = ys;
      const values = outputs.js() as number[];
      expect(Number.isFinite(values[0])).toBe(true);
      expect(values[0]).toBeGreaterThan(0.0);
    } finally {
      clearCaches();
      parEnv.viscosityWater.dispose();
      parEnv.densityWater.dispose();
      parPhotosynth.kmm.dispose();
      parPhotosynth.gammastar.dispose();
      parPhotosynth.phi0.dispose();
      parPhotosynth.Iabs.dispose();
      parPhotosynth.ca.dispose();
    }
  });
});
