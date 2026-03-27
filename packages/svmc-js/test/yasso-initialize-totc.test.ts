import { describe, it, expect } from "vitest";
import { numpy as baseNp, valueAndGrad } from "@hamk-uas/jax-js-nonconsuming";
import { initializeTotcFn } from "../src/yasso/index.js";
import { getNumericDType, np } from "../src/precision.js";
import yassoFixtures from "../../svmc-ref/fixtures/yasso.json";

const eps = baseNp.finfo(getNumericDType()).eps;

// Relative tolerance: initialize_totc involves matrix construction (~30 ops),
// linear solve (~50 ops), and CUE iteration (10 × ~15 ops).
// Total ~230 ops. Conservative 16384 * eps.
const RTOL = 16384 * eps;

type InitTotcCase = (typeof yassoFixtures.initialize_totc)[number];

function caseLabel(c: InitTotcCase, i: number): string {
  const inp = c.inputs;
  return `case ${i}: totc=${inp.totc}, legacy=${inp.fract_legacy_soc}, root=${inp.fract_root_input}, T=${inp.tempr_c}`;
}

describe("initialize_totc — Fortran reference", () => {
  for (const [i, c] of yassoFixtures.initialize_totc.entries()) {
    it(caseLabel(c, i), async () => {
      using param = np.array(c.inputs.param as number[]);
      using totc = np.array(c.inputs.totc as number);
      using cnInput = np.array(c.inputs.cn_input as number);
      using fractRootInput = np.array(c.inputs.fract_root_input as number);
      using fractLegacySoc = np.array(c.inputs.fract_legacy_soc as number);
      using temprC = np.array(c.inputs.tempr_c as number);
      using precipDay = np.array(c.inputs.precip_day as number);
      using temprAmpl = np.array(c.inputs.tempr_ampl as number);

      const { cstate, nstate } = initializeTotcFn(
        param, totc, cnInput, fractRootInput, fractLegacySoc,
        temprC, precipDay, temprAmpl,
      );
      using _cstate = cstate;
      using _nstate = nstate;

      expect(cstate).toBeAllclose(c.output.cstate as number[], { rtol: RTOL });
      using expectedNstate = np.array(c.output.nstate as number);
      expect(nstate).toBeAllclose(expectedNstate, { rtol: RTOL });
    });
  }
});

describe("initialize_totc — invariants", () => {
  it("carbon conservation: sum(cstate) == totc for all cases", async () => {
    for (const c of yassoFixtures.initialize_totc) {
      using param = np.array(c.inputs.param as number[]);
      using totc = np.array(c.inputs.totc as number);
      using cnInput = np.array(c.inputs.cn_input as number);
      using fractRootInput = np.array(c.inputs.fract_root_input as number);
      using fractLegacySoc = np.array(c.inputs.fract_legacy_soc as number);
      using temprC = np.array(c.inputs.tempr_c as number);
      using precipDay = np.array(c.inputs.precip_day as number);
      using temprAmpl = np.array(c.inputs.tempr_ampl as number);

      const { cstate, nstate } = initializeTotcFn(
        param, totc, cnInput, fractRootInput, fractLegacySoc,
        temprC, precipDay, temprAmpl,
      );
      using _cstate = cstate;
      using _nstate = nstate;
      using sumC = np.sum(cstate);
      expect(sumC).toBeAllclose(totc, { rtol: 1e-5 });
    }
  });

  it("full legacy: all carbon in H pool", async () => {
    const c = yassoFixtures.initialize_totc.find(
      (c) => c.inputs.fract_legacy_soc === 1.0,
    )!;
    using param = np.array(c.inputs.param as number[]);
    using totc = np.array(c.inputs.totc as number);
    using cnInput = np.array(c.inputs.cn_input as number);
    using fractRootInput = np.array(c.inputs.fract_root_input as number);
    using fractLegacySoc = np.array(c.inputs.fract_legacy_soc as number);
    using temprC = np.array(c.inputs.tempr_c as number);
    using precipDay = np.array(c.inputs.precip_day as number);
    using temprAmpl = np.array(c.inputs.tempr_ampl as number);

    const { cstate, nstate } = initializeTotcFn(
      param, totc, cnInput, fractRootInput, fractLegacySoc,
      temprC, precipDay, temprAmpl,
    );
    using _cstate = cstate;
    using _nstate = nstate;

    // AWEN pools should be zero, H pool should equal totc
    expect(cstate).toBeAllclose(
      [0, 0, 0, 0, c.inputs.totc as number],
      { atol: 1e-15 },
    );
  });

  it("leaf-root equivalence: awenh_leaf == awenh_fineroot", async () => {
    // Cases with fract_root=1.0 and fract_root=0.0 should match
    const rootCase = yassoFixtures.initialize_totc.find(
      (c) => c.inputs.fract_root_input === 1.0 && c.inputs.fract_legacy_soc === 0.0,
    )!;
    const leafCase = yassoFixtures.initialize_totc.find(
      (c) =>
        c.inputs.fract_root_input === 0.0 &&
        c.inputs.fract_legacy_soc === 0.0 &&
        c.inputs.totc === rootCase.inputs.totc,
    )!;
    expect(rootCase.output.cstate).toEqual(leafCase.output.cstate);
    expect(rootCase.output.nstate).toEqual(leafCase.output.nstate);
  });

  it("linearity: 10x totc gives 10x pools", async () => {
    const base = yassoFixtures.initialize_totc.find(
      (c) =>
        c.inputs.totc === 10.0 &&
        c.inputs.fract_legacy_soc === 0.0 &&
        c.inputs.tempr_c === 10.0,
    )!;
    const scaled = yassoFixtures.initialize_totc.find(
      (c) =>
        c.inputs.totc === 100.0 &&
        c.inputs.fract_legacy_soc === 0.0 &&
        c.inputs.tempr_c === 10.0,
    )!;
    const factor = scaled.inputs.totc / base.inputs.totc;
    for (let i = 0; i < 5; i++) {
      expect(scaled.output.cstate[i]).toBeCloseTo(
        (base.output.cstate[i] as number) * factor,
        4,
      );
    }
    expect(scaled.output.nstate).toBeCloseTo(
      (base.output.nstate as number) * factor,
      4,
    );
  });

  it("rejects fractRootInput outside [0, 1]", async () => {
    const c = yassoFixtures.initialize_totc[0];
    using param = np.array(c.inputs.param as number[]);
    using totc = np.array(c.inputs.totc as number);
    using cnInput = np.array(c.inputs.cn_input as number);
    using fractRootInput = np.array(1.1);
    using fractLegacySoc = np.array(c.inputs.fract_legacy_soc as number);
    using temprC = np.array(c.inputs.tempr_c as number);
    using precipDay = np.array(c.inputs.precip_day as number);
    using temprAmpl = np.array(c.inputs.tempr_ampl as number);

    expect(() => initializeTotcFn(
      param, totc, cnInput, fractRootInput, fractLegacySoc,
      temprC, precipDay, temprAmpl,
    )).toThrow(/fractRootInput must be in \[0, 1\]/);
  });

  it("rejects fractLegacySoc outside [0, 1]", async () => {
    const c = yassoFixtures.initialize_totc[0];
    using param = np.array(c.inputs.param as number[]);
    using totc = np.array(c.inputs.totc as number);
    using cnInput = np.array(c.inputs.cn_input as number);
    using fractRootInput = np.array(c.inputs.fract_root_input as number);
    using fractLegacySoc = np.array(-0.1);
    using temprC = np.array(c.inputs.tempr_c as number);
    using precipDay = np.array(c.inputs.precip_day as number);
    using temprAmpl = np.array(c.inputs.tempr_ampl as number);

    expect(() => initializeTotcFn(
      param, totc, cnInput, fractRootInput, fractLegacySoc,
      temprC, precipDay, temprAmpl,
    )).toThrow(/fractLegacySoc must be in \[0, 1\]/);
  });
});

describe("initialize_totc — autodiff", () => {
  // NOTE: jax-js-nonconsuming does not support reverse-mode differentiation
  // through np.where/np.minimum/np.maximum inside lax.foriLoop.
  // The CUE clamp in evalSteadystateNitr hits this limitation.
  // Forward-pass correctness is validated by the fixture playback tests above.
  it.skip("gradient w.r.t. totc (blocked: np.where inside foriLoop)", async () => {
    const c = yassoFixtures.initialize_totc[0];
    using param = np.array(c.inputs.param as number[]);
    using cnInput = np.array(c.inputs.cn_input as number);
    using fractRootInput = np.array(c.inputs.fract_root_input as number);
    using fractLegacySoc = np.array(c.inputs.fract_legacy_soc as number);
    using temprC = np.array(c.inputs.tempr_c as number);
    using precipDay = np.array(c.inputs.precip_day as number);
    using temprAmpl = np.array(c.inputs.tempr_ampl as number);

    const loss = (totc: np.Array) => {
      const { cstate, nstate } = initializeTotcFn(
        param, totc, cnInput, fractRootInput, fractLegacySoc,
        temprC, precipDay, temprAmpl,
      );
      using _nstate = nstate;
      using sumC = np.sum(cstate);
      return sumC.add(nstate);
    };

    const gradFn = valueAndGrad(loss);
    using totcInput = np.array(c.inputs.totc as number);
    const [value, grad] = gradFn(totcInput);

    expect(Number.isFinite(value.item())).toBe(true);
    using finiteGrad = np.isfinite(grad);
    using allFinite = np.all(finiteGrad);
    expect(allFinite.item()).toBe(1);
    value.dispose();
    grad.dispose();
  });
});
