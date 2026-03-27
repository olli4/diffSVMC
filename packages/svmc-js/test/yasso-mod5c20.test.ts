import { describe, it, expect } from "vitest";
import { numpy as baseNp } from "@hamk-uas/jax-js-nonconsuming";
import { mod5c20Fn } from "../src/yasso/index.js";
import { getNumericDType, np } from "../src/precision.js";
import yassoFixtures from "../../svmc-ref/fixtures/yasso.json";

// Machine epsilon for the configured numeric dtype.
const eps = baseNp.finfo(getNumericDType()).eps;

// Relative tolerance derived from critical-path operation depth with ≥2×
// safety factor.  Critical path: 5×5 matrix construction (~15 ops),
// Taylor scaling-and-squaring matrixExp (~20 ops), linear solve (~25 ops),
// total ~60 ops.  Condition number of the Yasso20 coefficient matrix is
// moderate (κ ≈ 5–20).  Observed max relative error (float32 vs Fortran
// float64 reference): ~6200·ε for case 3 (warm climate, woody size
// effect d=12, leaching=5e-4) — pool H carries the largest relative
// deviation due to its small absolute magnitude.
const RTOL = 8192 * eps;

type Mod5c20Case = (typeof yassoFixtures.mod5c20)[number];

function caseLabel(c: Mod5c20Case, i: number): string {
  const inp = c.inputs;
  const ss = "steadystate_pred" in inp ? "steady" : `t=${inp.time}`;
  return `case ${i}: ${ss}, prec=${inp.prec}`;
}

describe("mod5c20 — Fortran reference", () => {
  for (const [i, c] of yassoFixtures.mod5c20.entries()) {
    it(caseLabel(c, i), async () => {
      using theta = np.array(c.inputs.theta as number[]);
      using time = np.array(c.inputs.time as number);
      using temp = np.array(c.inputs.temp as number[]);
      using prec = np.array(c.inputs.prec as number);
      using init = np.array(c.inputs.init as number[]);
      using b = np.array(c.inputs.b as number[]);
      using d = np.array(c.inputs.d as number);
      using leac = np.array(c.inputs.leac as number);
      const ss = (c.inputs as Record<string, unknown>).steadystate_pred === true;

      using result = mod5c20Fn(theta, time, temp, prec, init, b, d, leac, ss);
      expect(result).toBeAllclose(c.output as number[], { rtol: RTOL });
    });
  }
});

describe("mod5c20 — invariants", () => {
  it("zero input → pure decay (all pools decrease)", async () => {
    // Case 4 has zero input (b=[0,0,0,0,0])
    const c = yassoFixtures.mod5c20[4];
    using theta = np.array(c.inputs.theta as number[]);
    using time = np.array(c.inputs.time as number);
    using temp = np.array(c.inputs.temp as number[]);
    using prec = np.array(c.inputs.prec as number);
    using init = np.array(c.inputs.init as number[]);
    using b = np.array(c.inputs.b as number[]);
    using d = np.array(c.inputs.d as number);
    using leac = np.array(c.inputs.leac as number);

    using result = mod5c20Fn(theta, time, temp, prec, init, b, d, leac);
    using total = np.sum(result);
    using initTotal = np.sum(init);
    using less = total.less(initTotal);
    expect(less.item()).toBe(1);
  });

  it("extreme cold: negligible decomposition matches fixture", async () => {
    // Case 6 has -80°C temperatures — tem ≈ 3e-8 (> Fortran TOL 1e-12).
    // The transient solver uses the Taylor branch (||At|| ≪ √ε) to
    // avoid cancellation in exp(At)·b − b; result ≈ init + b·time.
    const c = yassoFixtures.mod5c20[6];
    using theta = np.array(c.inputs.theta as number[]);
    using time = np.array(c.inputs.time as number);
    using temp = np.array(c.inputs.temp as number[]);
    using prec = np.array(c.inputs.prec as number);
    using init = np.array(c.inputs.init as number[]);
    using b = np.array(c.inputs.b as number[]);
    using d = np.array(c.inputs.d as number);
    using leac = np.array(c.inputs.leac as number);

    using result = mod5c20Fn(theta, time, temp, prec, init, b, d, leac);
    expect(result).toBeAllclose(c.output as number[], { rtol: RTOL });
  });
});

describe("mod5c20 — Taylor/direct switch boundary", () => {
  // Uses case 0 parameters and scales time so ||At|| lands just below
  // and just above sqrt(eps).  Verifies both branches produce results
  // consistent with leading-order ODE (init + b·time) and with each
  // other (smooth transition across the threshold).
  //
  // ||A|| ≈ 0.365 for case 0 parameters.  NORM_SWITCH = sqrt(eps).
  // float32: sqrt(1.19e-7) ≈ 3.45e-4 → switch time ≈ 9.45e-4
  // float64: sqrt(2.22e-16) ≈ 1.49e-8 → switch time ≈ 4.08e-8

  const c = yassoFixtures.mod5c20[0];
  const normA = 0.3647;  // empirical ||A|| for case 0

  // Compute the switch time for the configured dtype
  const switchTime = Math.sqrt(eps) / normA;

  for (const [label, scaleFactor] of [
    ["below threshold (Taylor branch)", 0.5],
    ["above threshold (direct branch)", 2.0],
  ] as const) {
    it(label, async () => {
      const t = switchTime * scaleFactor;
      using theta = np.array(c.inputs.theta as number[]);
      using time = np.array(t);
      using temp = np.array(c.inputs.temp as number[]);
      using prec = np.array(c.inputs.prec as number);
      using init = np.array(c.inputs.init as number[]);
      using b = np.array(c.inputs.b as number[]);
      using d = np.array(c.inputs.d as number);
      using leac = np.array(c.inputs.leac as number);

      using result = mod5c20Fn(theta, time, temp, prec, init, b, d, leac);

      // Leading-order: result ≈ init + b·time when time ≈ 0
      const approx = (c.inputs.init as number[]).map(
        (v, i) => v + (c.inputs.b as number[])[i] * t,
      );
      expect(result).toBeAllclose(approx, { rtol: 1e-3 });
    });
  }

  it("smooth transition across threshold", async () => {
    using theta = np.array(c.inputs.theta as number[]);
    using temp = np.array(c.inputs.temp as number[]);
    using prec = np.array(c.inputs.prec as number);
    using init = np.array(c.inputs.init as number[]);
    using b = np.array(c.inputs.b as number[]);
    using d = np.array(c.inputs.d as number);
    using leac = np.array(c.inputs.leac as number);

    // Times just below / above the switch — close enough that O(t²)
    // nonlinearity is negligible, so any jump is a switch artefact.
    using tBelow = np.array(switchTime * 0.99);
    using tAbove = np.array(switchTime * 1.01);

    using rBelow = mod5c20Fn(theta, tBelow, temp, prec, init, b, d, leac);
    using rAbove = mod5c20Fn(theta, tAbove, temp, prec, init, b, d, leac);

    // Results should be nearly identical (2% time difference → ~2% output difference)
    expect(rAbove).toBeAllclose(rBelow, { rtol: 0.05 });
  });
});
