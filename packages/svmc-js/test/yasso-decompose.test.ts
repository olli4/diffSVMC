import { describe, it, expect } from "vitest";
import { numpy as baseNp, valueAndGrad } from "@hamk-uas/jax-js-nonconsuming";
import { decomposeFn } from "../src/yasso/index.js";
import { getNumericDType, np } from "../src/precision.js";
import yassoFixtures from "../../svmc-ref/fixtures/yasso.json";

// Machine epsilon for the configured numeric dtype.
const eps = baseNp.finfo(getNumericDType()).eps;

// Relative tolerance: decompose has fewer ops than mod5c20 (no matrixexp,
// no linear solve), but still builds a 5x5 matrix and does matmul + N dynamics.
// Critical path: matrix construction (~15 ops), matmul (~5 ops), N dynamics (~15 ops).
// Total ~35 ops. Conservative 8192 * eps as with mod5c20.
const RTOL = 8192 * eps;

type DecomposeCase = (typeof yassoFixtures.decompose)[number];

function caseLabel(c: DecomposeCase, i: number): string {
  const inp = c.inputs;
  const totC = (inp.cstate as number[]).reduce((a, b) => a + b, 0);
  return `case ${i}: T=${inp.tempr_c}, P=${inp.precip_day}, totC=${totC.toExponential(1)}`;
}

function findDecomposeCase(predicate: (inputs: DecomposeCase["inputs"]) => boolean): DecomposeCase {
  const found = yassoFixtures.decompose.find((c) => predicate(c.inputs));
  if (!found) {
    throw new Error("Expected decompose fixture case was not found");
  }
  return found;
}

describe("decompose — Fortran reference", () => {
  for (const [i, c] of yassoFixtures.decompose.entries()) {
    it(caseLabel(c, i), async () => {
      using param = np.array(c.inputs.param as number[]);
      using timestepDays = np.array(c.inputs.timestep_days as number);
      using temprC = np.array(c.inputs.tempr_c as number);
      using precipDay = np.array(c.inputs.precip_day as number);
      using cstate = np.array(c.inputs.cstate as number[]);
      using nstate = np.array(c.inputs.nstate as number);

      const { ctend, ntend } = decomposeFn(
        param, timestepDays, temprC, precipDay, cstate, nstate,
      );
      using _ctend = ctend;
      using _ntend = ntend;

      expect(ctend).toBeAllclose(c.output.ctend as number[], { rtol: RTOL });
      using expectedNtend = np.array(c.output.ntend as number);
      expect(ntend).toBeAllclose(expectedNtend, { rtol: RTOL });
    });
  }
});

describe("decompose — invariants", () => {
  it("net carbon loss: sum(ctend) <= 0 for all cases", async () => {
    for (const c of yassoFixtures.decompose) {
      using param = np.array(c.inputs.param as number[]);
      using timestepDays = np.array(c.inputs.timestep_days as number);
      using temprC = np.array(c.inputs.tempr_c as number);
      using precipDay = np.array(c.inputs.precip_day as number);
      using cstate = np.array(c.inputs.cstate as number[]);
      using nstate = np.array(c.inputs.nstate as number);

      const { ctend, ntend } = decomposeFn(
        param, timestepDays, temprC, precipDay, cstate, nstate,
      );
      using _ctend = ctend;
      using _ntend = ntend;
      using total = np.sum(ctend);
      expect(total.item()).toBeLessThanOrEqual(0);
    }
  });

  it("temperature monotonicity: warmer → more respiration", async () => {
    // Cases 0-4 have same C state, varying temperature
    const cases = yassoFixtures.decompose.slice(0, 5);
    const respByTemp: Array<[number, number]> = [];

    for (const c of cases) {
      using param = np.array(c.inputs.param as number[]);
      using timestepDays = np.array(c.inputs.timestep_days as number);
      using temprC = np.array(c.inputs.tempr_c as number);
      using precipDay = np.array(c.inputs.precip_day as number);
      using cstate = np.array(c.inputs.cstate as number[]);
      using nstate = np.array(c.inputs.nstate as number);

      const { ctend, ntend } = decomposeFn(
        param, timestepDays, temprC, precipDay, cstate, nstate,
      );
      using _ctend = ctend;
      using _ntend = ntend;
      using negCtend = np.negative(ctend);
      using resp = np.sum(negCtend);
      respByTemp.push([c.inputs.tempr_c, resp.item() as number]);
    }

    respByTemp.sort((a, b) => a[0] - b[0]);
    for (let i = 0; i < respByTemp.length - 1; i++) {
      expect(respByTemp[i][1]).toBeLessThan(respByTemp[i + 1][1]);
    }
  });

  it("near-zero carbon yields ntend = 0", async () => {
    const c = findDecomposeCase((inp) =>
      (inp.cstate as number[]).reduce((a, b) => a + b, 0) < 1e-6);
    using param = np.array(c.inputs.param as number[]);
    using timestepDays = np.array(c.inputs.timestep_days as number);
    using temprC = np.array(c.inputs.tempr_c as number);
    using precipDay = np.array(c.inputs.precip_day as number);
    using cstate = np.array(c.inputs.cstate as number[]);
    using nstate = np.array(c.inputs.nstate as number);

    const { ctend, ntend } = decomposeFn(
      param, timestepDays, temprC, precipDay, cstate, nstate,
    );
    using _ctend = ctend;
    using _ntend = ntend;
    expect(ntend.item()).toBe(0);
  });

  it("explicit fixture triggers the unusual humus N:C branch", async () => {
    const c = findDecomposeCase((inp) =>
      inp.cstate[4] * 0.1 > inp.nstate
      && (inp.cstate as number[]).reduce((a, b) => a + b, 0) >= 1e-6);
    using param = np.array(c.inputs.param as number[]);
    using timestepDays = np.array(c.inputs.timestep_days as number);
    using temprC = np.array(c.inputs.tempr_c as number);
    using precipDay = np.array(c.inputs.precip_day as number);
    using cstate = np.array(c.inputs.cstate as number[]);
    using nstate = np.array(c.inputs.nstate as number);

    const { ctend, ntend } = decomposeFn(
      param, timestepDays, temprC, precipDay, cstate, nstate,
    );
    using _ctend = ctend;
    using _ntend = ntend;
    expect(Number.isFinite(ntend.item())).toBe(true);
  });

  it("explicit fixture triggers the CUE lower floor branch", async () => {
    const cueLowerThreshold = 0.008794663773278361;
    const c = findDecomposeCase((inp) => {
      const totC = (inp.cstate as number[]).reduce((a, b) => a + b, 0);
      return totC >= 1e-6
        && inp.cstate[4] * 0.1 <= inp.nstate
        && inp.nstate / totC < cueLowerThreshold;
    });
    using param = np.array(c.inputs.param as number[]);
    using timestepDays = np.array(c.inputs.timestep_days as number);
    using temprC = np.array(c.inputs.tempr_c as number);
    using precipDay = np.array(c.inputs.precip_day as number);
    using cstate = np.array(c.inputs.cstate as number[]);
    using nstate = np.array(c.inputs.nstate as number);

    const { ctend, ntend } = decomposeFn(
      param, timestepDays, temprC, precipDay, cstate, nstate,
    );
    using _ctend = ctend;
    using _ntend = ntend;
    expect(Number.isFinite(ntend.item())).toBe(true);
  });

  it("ntend is finite for all cases", async () => {
    for (const c of yassoFixtures.decompose) {
      using param = np.array(c.inputs.param as number[]);
      using timestepDays = np.array(c.inputs.timestep_days as number);
      using temprC = np.array(c.inputs.tempr_c as number);
      using precipDay = np.array(c.inputs.precip_day as number);
      using cstate = np.array(c.inputs.cstate as number[]);
      using nstate = np.array(c.inputs.nstate as number);

      const { ctend, ntend } = decomposeFn(
        param, timestepDays, temprC, precipDay, cstate, nstate,
      );
      using _ctend = ctend;
      using _ntend = ntend;
      expect(Number.isFinite(ntend.item())).toBe(true);
    }
  });

  it("decompose remains differentiable w.r.t. cstate", async () => {
    const c = yassoFixtures.decompose[0];
    using param = np.array(c.inputs.param as number[]);
    using timestepDays = np.array(c.inputs.timestep_days as number);
    using temprC = np.array(c.inputs.tempr_c as number);
    using precipDay = np.array(c.inputs.precip_day as number);
    using nstate = np.array(c.inputs.nstate as number);

    const loss = (cs: np.Array) => {
      const { ctend, ntend } = decomposeFn(param, timestepDays, temprC, precipDay, cs, nstate);
      using _ctend = ctend;
      using _ntend = ntend;
      using sumCtend = np.sum(ctend);
      return sumCtend.add(ntend);
    };

    const gradFn = valueAndGrad(loss);
    using cstate = np.array(c.inputs.cstate as number[]);
    const [value, grad] = gradFn(cstate);
    expect(Number.isFinite(value.item())).toBe(true);
    using finiteGrad = np.isfinite(grad);
    using allFiniteGrad = np.all(finiteGrad);
    expect(allFiniteGrad.item()).toBe(1);
    value.dispose();
    grad.dispose();
  });
});
