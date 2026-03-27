import { describe, expect, it } from "vitest";
import { DType, numpy as baseNp } from "@hamk-uas/jax-js-nonconsuming";
import { canopyWaterSnow } from "../src/water/index.js";
import type { CanopySnowParams, CanopyWaterState } from "../src/water/canopy-soil.js";
import { createPrecisionNp, getNumericDType, np } from "../src/precision.js";
import waterFixtures from "../../svmc-ref/fixtures/water.json";

function absDiff(actual: number, expected: number): number {
  return Math.abs(actual - expected);
}

function runCanopyWaterSnowFixture() {
  const fixture = waterFixtures.canopy_water_snow[6];
  const { inputs: inp, output: exp } = fixture;

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
    gsoil: np.array(0.01),
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
    const mbeError = absDiff(flux.mbe.js() as number, exp.mbe);
    const totalAbsError =
      absDiff(newState.CanopyStorage.js() as number, exp.CanopyStorage)
      + absDiff(newState.SWE.js() as number, exp.SWE)
      + absDiff(newState.swe_i.js() as number, exp.swe_i)
      + absDiff(newState.swe_l.js() as number, exp.swe_l)
      + absDiff(flux.Throughfall.js() as number, exp.Throughfall)
      + absDiff(flux.Interception.js() as number, exp.Interception)
      + absDiff(flux.CanopyEvap.js() as number, exp.CanopyEvap)
      + absDiff(flux.Unloading.js() as number, exp.Unloading)
      + absDiff(flux.PotInfiltration.js() as number, exp.PotInfiltration)
      + absDiff(flux.Melt.js() as number, exp.Melt)
      + absDiff(flux.Freeze.js() as number, exp.Freeze)
      + mbeError;

    return { mbeError, totalAbsError };
  } finally {
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
}

describe("svmc-js precision configuration", () => {
  it("creates precision-bound np namespaces without mutable global state", () => {
    const np32 = createPrecisionNp(DType.Float32);
    const np64 = createPrecisionNp(DType.Float64);
    using f32 = np32.array(1.25);
    using f64 = np64.array(1.25);
    using bools = np64.array([true, false]);

    expect(f32.dtype).toBe(DType.Float32);
    expect(f64.dtype).toBe(DType.Float64);
    expect(bools.dtype).toBe(DType.Bool);
  });

  it("respects the configured dtype for default numeric arrays", () => {
    const configured = getNumericDType();
    using value = np.array(1.25);
    expect(value.dtype).toBe(configured);
  });

  it("keeps canopy_water_snow fixture error within an epsilon-scaled bound", () => {
    const result = runCanopyWaterSnowFixture();
    const dtype = getNumericDType();
    const eps = baseNp.finfo(dtype).eps;
    const fixture = waterFixtures.canopy_water_snow[6];
    const expectedScale =
      Math.abs(fixture.output.CanopyStorage)
      + Math.abs(fixture.output.SWE)
      + Math.abs(fixture.output.swe_i)
      + Math.abs(fixture.output.swe_l)
      + Math.abs(fixture.output.Throughfall)
      + Math.abs(fixture.output.Interception)
      + Math.abs(fixture.output.CanopyEvap)
      + Math.abs(fixture.output.Unloading)
      + Math.abs(fixture.output.PotInfiltration)
      + Math.abs(fixture.output.Melt)
      + Math.abs(fixture.output.Freeze)
      + Math.abs(fixture.output.mbe)
      + 1.0;
    // canopy_water_snow has a 14-operation critical path (phase fractions,
    // exponential interception, PM × 2, melt/freeze, state updates) with ~90
    // total arithmetic operations.  The budget uses the total op count with a
    // 2× safety factor: ceil(90 × 2) = 192.  The "+12" addend covers the 12
    // output fields each contributing up to eps absolute noise to the
    // totalAbsError sum.
    const opsDepth = 192;
    const outputFields = 12;
    const accumulationBudget = opsDepth * eps * expectedScale + outputFields * eps;

    expect(result.totalAbsError).toBeLessThan(accumulationBudget);
    expect(result.mbeError).toBeLessThan(accumulationBudget);
  });
});