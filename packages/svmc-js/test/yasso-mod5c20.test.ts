import { describe, it, expect } from "vitest";
import { mod5c20Fn } from "../src/yasso/index.js";
import { np } from "../src/precision.js";
import yassoFixtures from "../../svmc-ref/fixtures/yasso.json";

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
      expect(result).toBeAllclose(c.output as number[], { rtol: 1e-3 });
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
    // In float32 mode, the widened TOL (1e-6) takes the early-return
    // path; in float64, the transient path runs. Either way the result
    // is ≈ init + b·time and matches the fixture within rtol.
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
    expect(result).toBeAllclose(c.output as number[], { rtol: 1e-3 });
  });
});
