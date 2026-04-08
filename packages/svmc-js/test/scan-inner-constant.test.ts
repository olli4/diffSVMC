/**
 * Minimal reproduction tests for the hypothesis that creating fresh constants
 * (e.g. np.array(0)) inside a traced lax.foriLoop body nested within lax.scan
 * causes "Referenced tracer Array has been disposed" errors.
 *
 * These tests isolate the jax-js-nonconsuming library behavior without any
 * physics code, to determine whether the library can handle dynamically
 * created constants inside traced sub-graphs.
 */
import { describe, expect, it } from "vitest";
import { grad, hessian, lax, type JsTree } from "@hamk-uas/jax-js-nonconsuming";
import { np } from "../src/precision.js";

describe("scan + foriLoop inner constant handling", () => {
  // -----------------------------------------------------------------------
  // Baseline: simple scan without nested foriLoop — should work
  // -----------------------------------------------------------------------
  it("scan alone works with inner constant", () => {
    using init = np.array(0.0);
    using xs = np.ones([3]);

    const [finalCarry, ys] = lax.scan(
      (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        // Create a fresh constant inside the traced scan body
        const one = np.array(1.0);
        return [carry.add(one), x.mul(carry)];
      },
      init,
      xs,
      { length: 3 },
    );

    using fc = finalCarry;
    using y = ys;
    expect(fc.js()).toBeCloseTo(3.0);
  });

  // -----------------------------------------------------------------------
  // Baseline: foriLoop alone — should work
  // -----------------------------------------------------------------------
  it("foriLoop alone works with inner constant", () => {
    using init = np.array(0.0);

    const result = lax.foriLoop(
      0,
      5,
      (i: np.Array, carry: np.Array): np.Array => {
        // Create a fresh constant inside the traced foriLoop body
        const two = np.array(2.0);
        return carry.add(two);
      },
      init,
    );

    using r = result as np.Array;
    expect(r.js()).toBeCloseTo(10.0);
  });

  // -----------------------------------------------------------------------
  // THE MAIN HYPOTHESIS: nested foriLoop inside scan with inner constant
  // -----------------------------------------------------------------------
  it("scan + nested foriLoop with inner constant in foriLoop body", () => {
    using init = np.array(0.0);
    using xs = np.ones([2]);

    const [finalCarry, ys] = lax.scan(
      (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        // Nested foriLoop that creates a fresh constant on each iteration
        const innerResult = lax.foriLoop(
          0,
          3,
          (i: np.Array, innerCarry: np.Array): np.Array => {
            const step = np.array(1.0); // fresh constant inside nested loop
            return innerCarry.add(step);
          },
          carry,
        ) as np.Array;

        return [innerResult, innerResult.mul(x)];
      },
      init,
      xs,
      { length: 2 },
    );

    using fc = finalCarry;
    using y = ys;
    // carry: 0 -> +3 -> +3 = 6
    expect(fc.js()).toBeCloseTo(6.0);
  });

  // -----------------------------------------------------------------------
  // Variant: inner constant in scan body, but NOT inside foriLoop
  // -----------------------------------------------------------------------
  it("scan + nested foriLoop with inner constant only in scan body", () => {
    using init = np.array(0.0);
    using xs = np.ones([2]);

    const [finalCarry, ys] = lax.scan(
      (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        // Constant created in the scan body, passed into foriLoop
        const step = np.array(1.0);

        const innerResult = lax.foriLoop(
          0,
          3,
          (i: np.Array, innerCarry: np.Array): np.Array => {
            return innerCarry.add(step); // uses closure-captured constant
          },
          carry,
        ) as np.Array;

        return [innerResult, innerResult.mul(x)];
      },
      init,
      xs,
      { length: 2 },
    );

    using fc = finalCarry;
    using y = ys;
    expect(fc.js()).toBeCloseTo(6.0);
  });

  // -----------------------------------------------------------------------
  // Variant: constant created outside both loops (the "safe" pattern)
  // -----------------------------------------------------------------------
  it("scan + nested foriLoop with constant outside both loops", () => {
    using init = np.array(0.0);
    using xs = np.ones([2]);
    using step = np.array(1.0); // constant created OUTSIDE all traced scopes

    const [finalCarry, ys] = lax.scan(
      (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const innerResult = lax.foriLoop(
          0,
          3,
          (i: np.Array, innerCarry: np.Array): np.Array => {
            return innerCarry.add(step); // closure-captured from outer scope
          },
          carry,
        ) as np.Array;

        return [innerResult, innerResult.mul(x)];
      },
      init,
      xs,
      { length: 2 },
    );

    using fc = finalCarry;
    using y = ys;
    expect(fc.js()).toBeCloseTo(6.0);
  });

  // -----------------------------------------------------------------------
  // Variant: derive zero from the carry instead of np.array(0)
  // (this is the "topology fix" we tried in soilWater)
  // -----------------------------------------------------------------------
  it("scan + nested foriLoop deriving zero from carry", () => {
    using init = np.array(0.0);
    using xs = np.ones([2]);

    const [finalCarry, ys] = lax.scan(
      (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const innerResult = lax.foriLoop(
          0,
          3,
          (i: np.Array, innerCarry: np.Array): np.Array => {
            const step = innerCarry.mul(0.0).add(1.0); // derive from carry
            return innerCarry.add(step);
          },
          carry,
        ) as np.Array;

        return [innerResult, innerResult.mul(x)];
      },
      init,
      xs,
      { length: 2 },
    );

    using fc = finalCarry;
    using y = ys;
    expect(fc.js()).toBeCloseTo(6.0);
  });

  // -----------------------------------------------------------------------
  // Structured carry (pytree) — closer to the real soilWater pattern
  // -----------------------------------------------------------------------
  it("scan + nested foriLoop with structured carry and inner constants", () => {
    using initA = np.array(0.0);
    using initB = np.array(1.0);
    const init = { a: initA, b: initB };
    using xs = np.ones([2]);

    type Carry = { a: np.Array; b: np.Array };

    const [finalCarry, ys] = lax.scan(
      (carry: Carry, x: np.Array): [Carry, np.Array] => {
        const innerResult = lax.foriLoop(
          0,
          3,
          (i: np.Array, innerCarry: Carry): Carry => {
            const delta = np.array(0.5);  // fresh scalar constant
            const zero = np.array(0.0);   // fresh zero constant
            return {
              a: innerCarry.a.add(delta),
              b: innerCarry.b.add(zero),
            };
          },
          carry,
        ) as unknown as Carry;

        return [innerResult, innerResult.a.add(innerResult.b)];
      },
      init as unknown as JsTree<np.Array>,
      xs as unknown as JsTree<np.Array>,
      { length: 2 },
    ) as unknown as [Carry, np.Array];

    using fa = finalCarry.a;
    using fb = finalCarry.b;
    using y = ys;
    // a: 0 -> +1.5 -> +1.5 = 3.0
    // b: 1 -> +0   -> +0   = 1.0
    expect(fa.js()).toBeCloseTo(3.0);
    expect(fb.js()).toBeCloseTo(1.0);
  });

  // -----------------------------------------------------------------------
  // Multiple fresh constants — stress test
  // -----------------------------------------------------------------------
  it("scan + nested foriLoop with many inner constants", () => {
    using init = np.array(0.0);
    using xs = np.ones([2]);

    const [finalCarry, ys] = lax.scan(
      (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const innerResult = lax.foriLoop(
          0,
          3,
          (i: np.Array, innerCarry: np.Array): np.Array => {
            // Multiple independent constants created in each iteration
            const a = np.array(1.0);
            const b = np.array(2.0);
            const c = np.array(0.5);
            const combined = a.add(b).mul(c); // = 1.5
            return innerCarry.add(combined);
          },
          carry,
        ) as np.Array;

        return [innerResult, innerResult.mul(x)];
      },
      init,
      xs,
      { length: 2 },
    );

    using fc = finalCarry;
    using y = ys;
    // carry: 0 -> +4.5 -> +4.5 = 9.0
    expect(fc.js()).toBeCloseTo(9.0);
  });

  // -----------------------------------------------------------------------
  // np.zeros / np.ones inside the inner loop
  // -----------------------------------------------------------------------
  it("scan + nested foriLoop with np.zeros inside inner body", () => {
    using init = np.array(5.0);
    using xs = np.ones([2]);

    const [finalCarry, ys] = lax.scan(
      (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const innerResult = lax.foriLoop(
          0,
          2,
          (i: np.Array, innerCarry: np.Array): np.Array => {
            const zero = np.zeros([]); // np.zeros inside inner body
            const addend = zero.add(1.0);
            return innerCarry.add(addend);
          },
          carry,
        ) as np.Array;

        return [innerResult, innerResult];
      },
      init,
      xs,
      { length: 2 },
    );

    using fc = finalCarry;
    using y = ys;
    // carry: 5 -> +2 -> +2 = 9
    expect(fc.js()).toBeCloseTo(9.0);
  });

  // -----------------------------------------------------------------------
  // Pattern matching soilWater: helper function called from within foriLoop
  // that creates constants internally
  // -----------------------------------------------------------------------
  it("scan + nested foriLoop calling helper function that creates constants", () => {
    // Simulates calling soilWater / canopyWaterFlux from inside the hourly loop
    function bucketUpdate(state: np.Array, input: np.Array): np.Array {
      const capacity = np.array(10.0);  // like watsat
      const residual = np.array(0.1);   // like watres
      const rate = np.array(0.5);       // like ksat
      const delta = input.mul(rate);
      const newState = state.add(delta);
      // Clamp to [residual, capacity]
      const clamped = np.maximum(residual, np.minimum(capacity, newState));
      return clamped;
    }

    using init = np.array(5.0);
    using xs = np.ones([2]);

    const [finalCarry, ys] = lax.scan(
      (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const innerResult = lax.foriLoop(
          0,
          3,
          (i: np.Array, innerCarry: np.Array): np.Array => {
            return bucketUpdate(innerCarry, x);
          },
          carry,
        ) as np.Array;

        return [innerResult, innerResult];
      },
      init,
      xs,
      { length: 2 },
    );

    using fc = finalCarry;
    using y = ys;
    // bucketUpdate adds 0.5 each step, clamped to [0.1, 10.0]
    // 5 -> 5.5 -> 6.0 -> 6.5  (day 1 after 3 hours)
    // 6.5 -> 7.0 -> 7.5 -> 8.0 (day 2 after 3 hours)
    expect(fc.js()).toBeCloseTo(8.0);
  });

  // -----------------------------------------------------------------------
  // Pattern: closure-captured parameter objects (like soilParams, aeroParams)
  // -----------------------------------------------------------------------
  it("scan + nested foriLoop with closure-captured param objects", () => {
    // Simulate capturing soilParams/aeroParams from outer scope
    using paramA = np.array(0.5);
    using paramB = np.array(2.0);
    const params = { a: paramA, b: paramB };

    using init = np.array(1.0);
    using xs = np.ones([2]);

    const [finalCarry, ys] = lax.scan(
      (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const innerResult = lax.foriLoop(
          0,
          3,
          (i: np.Array, innerCarry: np.Array): np.Array => {
            // Access closure-captured params + create inner constant
            const scale = np.array(0.1);
            return innerCarry.add(params.a.mul(scale)).add(params.b.mul(x));
          },
          carry,
        ) as np.Array;

        return [innerResult, innerResult];
      },
      init,
      xs,
      { length: 2 },
    );

    using fc = finalCarry;
    using y = ys;
    // Each inner iteration: carry += 0.5*0.1 + 2.0*1.0 = 2.05
    // After 3 iters: 1.0 + 6.15 = 7.15
    // After 6 iters: 7.15 + 6.15 = 13.30
    expect(fc.js()).toBeCloseTo(13.3);
  });

  // -----------------------------------------------------------------------
  // Pattern: dynamicSlice inside foriLoop inside scan (hourly forcing pattern)
  // -----------------------------------------------------------------------
  it("scan + nested foriLoop with dynamicSlice on captured array", () => {
    using init = np.array(0.0);
    using xs = np.ones([2]);
    using hourlyData = np.array([1.0, 2.0, 3.0, 4.0]); // simulate hourly forcing

    const [finalCarry, ys] = lax.scan(
      (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const innerResult = lax.foriLoop(
          0,
          4,
          (i: np.Array, innerCarry: np.Array): np.Array => {
            const val = lax.dynamicSlice(hourlyData, [i], [1]).reshape([]);
            return innerCarry.add(val);
          },
          carry,
        ) as np.Array;

        return [innerResult, innerResult];
      },
      init,
      xs,
      { length: 2 },
    );

    using fc = finalCarry;
    using y = ys;
    // sum(1+2+3+4) = 10 per day, 2 days = 20
    expect(fc.js()).toBeCloseTo(20.0);
  });

  // -----------------------------------------------------------------------
  // Pattern: structured carry + helper functions + dynamicSlice + params
  // This is the closest match to the real integration code
  // -----------------------------------------------------------------------
  it("scan + nested foriLoop full integration pattern", () => {
    type State = { water: np.Array; snow: np.Array };

    using waterInit = np.array(5.0);
    using snowInit = np.array(0.0);
    const init: State = { water: waterInit, snow: snowInit };

    using xs = np.ones([2]);
    using hourlyTemp = np.array([270.0, 271.0, 273.0, 275.0]);
    using hourlyPrec = np.array([0.1, 0.2, 0.0, 0.3]);

    using capacity = np.array(10.0);
    using meltRate = np.array(0.01);
    const params = { capacity, meltRate };

    function waterBalance(
      state: State,
      temp: np.Array,
      prec: np.Array,
      params: { capacity: np.Array; meltRate: np.Array },
    ): State {
      const freezeThreshold = np.array(273.15);  // constant inside helper
      const isWarm = temp.sub(freezeThreshold);   // negative = freezing
      const snowMelt = np.maximum(np.array(0.0), state.snow.mul(params.meltRate).mul(isWarm));
      const newSnow = np.maximum(np.array(0.0), prec.mul(np.minimum(np.array(1.0), freezeThreshold.sub(temp))));
      const rain = prec.sub(newSnow);
      const waterInput = rain.add(snowMelt);
      const newWater = np.minimum(params.capacity, state.water.add(waterInput));
      const newSnowState = np.maximum(np.array(0.0), state.snow.add(newSnow).sub(snowMelt));
      return { water: newWater, snow: newSnowState };
    }

    const [finalCarry, ys] = lax.scan(
      (carry: State, x: np.Array): [State, np.Array] => {
        const hourlyResult = lax.foriLoop(
          0,
          4,
          (i: np.Array, hourlyCarry: State): State => {
            const temp = lax.dynamicSlice(hourlyTemp, [i], [1]).reshape([]);
            const prec = lax.dynamicSlice(hourlyPrec, [i], [1]).reshape([]);
            return waterBalance(hourlyCarry, temp, prec, params);
          },
          carry,
        ) as unknown as State;

        return [hourlyResult, hourlyResult.water];
      },
      init as unknown as JsTree<np.Array>,
      xs as unknown as JsTree<np.Array>,
      { length: 2 },
    ) as unknown as [State, np.Array];

    using fw = finalCarry.water;
    using fs = finalCarry.snow;
    using y = ys;
    // Just verify it runs without disposed-tracer errors and produces finite values
    expect(Number.isFinite(fw.js() as number)).toBe(true);
    expect(Number.isFinite(fs.js() as number)).toBe(true);
    expect((fw.js() as number)).toBeGreaterThan(0);
  });

  // -----------------------------------------------------------------------
  // Triple nesting: scan → foriLoop → foriLoop (like solver inside hourly)
  // -----------------------------------------------------------------------
  it("scan + foriLoop + nested foriLoop (triple nesting)", () => {
    using init = np.array(0.0);
    using xs = np.ones([2]);

    const [finalCarry, ys] = lax.scan(
      (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        // Outer foriLoop (hourly)
        const outerResult = lax.foriLoop(
          0,
          3,
          (i: np.Array, outerCarry: np.Array): np.Array => {
            // Inner foriLoop (solver iterations)
            const solverResult = lax.foriLoop(
              0,
              2,
              (j: np.Array, solverCarry: np.Array): np.Array => {
                const step = np.array(0.1);
                return solverCarry.add(step);
              },
              outerCarry,
            ) as np.Array;
            return solverResult;
          },
          carry,
        ) as np.Array;

        return [outerResult, outerResult];
      },
      init,
      xs,
      { length: 2 },
    );

    using fc = finalCarry;
    using y = ys;
    // 2 scan iters × 3 outer × 2 inner × 0.1 = 1.2
    expect(fc.js()).toBeCloseTo(1.2);
  });

  // -----------------------------------------------------------------------
  // THE KEY PATTERN: grad() inside foriLoop inside scan
  // This is what the P-Hydro solver does
  // -----------------------------------------------------------------------
  it("scan + foriLoop with grad() inside inner loop body", () => {
    using init = np.array(1.0);
    using xs = np.ones([2]);

    const [finalCarry, ys] = lax.scan(
      (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const innerResult = lax.foriLoop(
          0,
          3,
          (i: np.Array, innerCarry: np.Array): np.Array => {
            // Objective function: f(x) = (x - 2)^2
            const objective = (x: np.Array): np.Array => {
              const target = np.array(2.0);
              const diff = x.sub(target);
              return diff.mul(diff);
            };
            // Compute gradient: df/dx = 2(x - 2)
            const gradFn = grad(objective);
            const g = gradFn(innerCarry);
            // Gradient descent step
            const stepSize = np.array(0.1);
            return innerCarry.sub(g.mul(stepSize));
          },
          carry,
        ) as np.Array;

        return [innerResult, innerResult];
      },
      init,
      xs,
      { length: 2 },
    );

    using fc = finalCarry;
    using y = ys;
    // Gradient descent toward 2.0
    expect(Number.isFinite(fc.js() as number)).toBe(true);
    expect((fc.js() as number)).toBeGreaterThan(1.0);
    expect((fc.js() as number)).toBeLessThan(2.5);
  });

  // -----------------------------------------------------------------------
  // grad() + hessian() inside triple nesting (exact solver pattern)
  // -----------------------------------------------------------------------
  it("scan + foriLoop + grad + hessian (Newton-like inner solver)", () => {
    using init = np.array(0.5);
    using xs = np.ones([2]);

    const [finalCarry, ys] = lax.scan(
      (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        // Hourly foriLoop
        const hourlyResult = lax.foriLoop(
          0,
          2,
          (hourIdx: np.Array, hourlyCarry: np.Array): np.Array => {
            // Newton solver foriLoop (like projectedNewtonSolveImpl)
            const solverResult = lax.foriLoop(
              0,
              3,
              (solverIdx: np.Array, solverCarry: np.Array): np.Array => {
                const objective = (x: np.Array): np.Array => {
                  const target = np.array(3.0);
                  const diff = x.sub(target);
                  return diff.mul(diff);
                };
                const g = grad(objective)(solverCarry);
                const stepSize = np.array(0.1);
                return solverCarry.sub(g.mul(stepSize));
              },
              hourlyCarry,
            ) as np.Array;
            return solverResult;
          },
          carry,
        ) as np.Array;

        return [hourlyResult, hourlyResult];
      },
      init,
      xs,
      { length: 2 },
    );

    using fc = finalCarry;
    using y = ys;
    // Should converge toward 3.0
    expect(Number.isFinite(fc.js() as number)).toBe(true);
    expect((fc.js() as number)).toBeGreaterThan(0.5);
  });

  // -----------------------------------------------------------------------
  // Structured carry with grad in triple nesting (closest to real solver)
  // -----------------------------------------------------------------------
  it("scan + foriLoop + solver-like structured carry with grad", () => {
    type SolverCarry = { x: np.Array; lam: np.Array };

    using xInit = np.array(1.0);
    using lamInit = np.array(0.1);
    const init: SolverCarry = { x: xInit, lam: lamInit };
    using xs = np.ones([2]);

    const [finalCarry, ys] = lax.scan(
      (carry: SolverCarry, _x: np.Array): [SolverCarry, np.Array] => {
        const hourlyResult = lax.foriLoop(
          0,
          2,
          (hourIdx: np.Array, hourlyCarry: SolverCarry): SolverCarry => {
            // Newton-like solver
            const solverResult = lax.foriLoop(
              0,
              3,
              (solverIdx: np.Array, sc: SolverCarry): SolverCarry => {
                const objective = (x: np.Array): np.Array => {
                  const target = np.array(2.0);
                  return x.sub(target).mul(x.sub(target));
                };
                const g = grad(objective)(sc.x);
                const newX = sc.x.sub(g.mul(sc.lam));
                const improved = objective(newX).less(objective(sc.x));
                const nextX = np.where(improved, newX, sc.x);
                const shrink = sc.lam.mul(0.9);
                const grow = sc.lam.mul(1.1);
                const nextLam = np.where(improved, shrink, grow);
                return { x: nextX, lam: nextLam };
              },
              hourlyCarry,
            ) as unknown as SolverCarry;
            return solverResult;
          },
          carry,
        ) as unknown as SolverCarry;

        return [hourlyResult, hourlyResult.x];
      },
      init as unknown as JsTree<np.Array>,
      xs as unknown as JsTree<np.Array>,
      { length: 2 },
    ) as unknown as [SolverCarry, np.Array];

    using fx = finalCarry.x;
    using fl = finalCarry.lam;
    using y = ys;
    expect(Number.isFinite(fx.js() as number)).toBe(true);
    expect(Number.isFinite(fl.js() as number)).toBe(true);
  });
});
