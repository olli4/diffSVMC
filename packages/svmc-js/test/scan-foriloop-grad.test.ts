/**
 * Minimal reproduction of the disposed-tracer error:
 *   scan → foriLoop → grad
 * 
 * Hypothesis: During grad's PE evaluation inside foriLoop's makeJaxpr,
 * `bind()` of all-known ops goes through the foriLoop's JaxprTrace
 * (not the PE trace) because `dynamicTrace` leaks from the surrounding scope.
 * This may cause ref/dispose imbalances on concrete scalar params.
 */
import { describe, expect, it } from "vitest";
import { grad, hessian, lax, tree, type JsTree } from "@hamk-uas/jax-js-nonconsuming";
import { np } from "../src/precision.js";
import { pmodelHydraulicsNumerical } from "../src/phydro/index.js";

describe("scan → foriLoop → grad minimal repro", () => {
  it("scalar param survives through nested grad", () => {
    // Scalar param created OUTSIDE any trace (owner=none)
    using alpha = np.array(2.5);
    using beta = np.array(0.3);

    type Carry = { x: np.Array; y: np.Array };

    const init: Carry = { x: np.array(1.0), y: np.array(0.0) };
    using xs = np.array([0.0]); // 1 scan step

    let result: Carry | null = null;
    let ys: np.Array | null = null;

    try {
      const scanBody = (carry: Carry, _x: np.Array): [Carry, np.Array] => {
        // foriLoop inside scan body
        const hourlyBody = (i: np.Array, c: Carry): Carry => {
          // grad inside foriLoop body — this triggers nested PE + bind
          const lossFn = (v: np.Array): np.Array => {
            // Use closure-captured scalar params (alpha, beta)
            return v.mul(alpha).add(beta).mul(v);
          };
          const g = grad(lossFn)(c.x);
          return { x: c.x.add(g.mul(0.01)), y: c.y.add(g) };
        };

        const inner = lax.foriLoop(
          0, 2,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          carry as unknown as JsTree<np.Array>,
        ) as unknown as Carry;

        return [inner, inner.y];
      };

      [result as unknown, ys] = lax.scan(
        scanBody as unknown as (c: JsTree<np.Array>, x: JsTree<np.Array>) => [JsTree<np.Array>, JsTree<np.Array>],
        init as unknown as JsTree<np.Array>,
        xs as unknown as JsTree<np.Array>,
        { length: 1 },
      );

      expect(Number.isFinite((result!.x as np.Array).js() as number)).toBe(true);
    } finally {
      if (ys != null) (ys as np.Array).dispose();
      if (result != null) tree.dispose(result);
      tree.dispose(init);
    }
  });

  it("triple nesting: scan → foriLoop → foriLoop → grad", () => {
    // This matches the actual pattern: scan → hourlyLoop → newtonLoop → grad
    using alpha = np.array(2.5);
    using beta = np.array(0.3);

    type Carry = { x: np.Array; y: np.Array };

    const init: Carry = { x: np.array(1.0), y: np.array(0.0) };
    using xs = np.array([0.0]);

    let result: Carry | null = null;
    let ys: np.Array | null = null;

    try {
      const scanBody = (carry: Carry, _x: np.Array): [Carry, np.Array] => {
        // Outer foriLoop (hourly)
        const hourlyBody = (i: np.Array, c: Carry): Carry => {
          // Inner foriLoop (newton solver) with grad inside
          const solverBody = (j: np.Array, state: np.Array): np.Array => {
            const lossFn = (v: np.Array): np.Array => {
              return v.mul(alpha).add(beta).mul(v);
            };
            const g = grad(lossFn)(state);
            return state.sub(g.mul(0.01));
          };

          const solved = lax.foriLoop(0, 3, solverBody, c.x);
          return { x: solved, y: c.y.add(solved) };
        };

        const inner = lax.foriLoop(
          0, 2,
          hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
          carry as unknown as JsTree<np.Array>,
        ) as unknown as Carry;

        return [inner, inner.y];
      };

      [result as unknown, ys] = lax.scan(
        scanBody as unknown as (c: JsTree<np.Array>, x: JsTree<np.Array>) => [JsTree<np.Array>, JsTree<np.Array>],
        init as unknown as JsTree<np.Array>,
        xs as unknown as JsTree<np.Array>,
        { length: 1 },
      );

      expect(Number.isFinite((result!.x as np.Array).js() as number)).toBe(true);
    } finally {
      if (ys != null) (ys as np.Array).dispose();
      if (result != null) tree.dispose(result);
      tree.dispose(init);
    }
  });

  it("scalar param survives through nested grad (no scan)", () => {
    // Same test but without scan wrapper — just foriLoop → grad
    using alpha = np.array(2.5);
    using beta = np.array(0.3);

    type Carry = { x: np.Array; y: np.Array };

    const init: Carry = { x: np.array(1.0), y: np.array(0.0) };

    let result: Carry | null = null;
    try {
      const hourlyBody = (i: np.Array, c: Carry): Carry => {
        const lossFn = (v: np.Array): np.Array => {
          return v.mul(alpha).add(beta).mul(v);
        };
        const g = grad(lossFn)(c.x);
        return { x: c.x.add(g.mul(0.01)), y: c.y.add(g) };
      };

      result = lax.foriLoop(
        0, 2,
        hourlyBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
        init as unknown as JsTree<np.Array>,
      ) as unknown as Carry;

      expect(Number.isFinite((result!.x as np.Array).js() as number)).toBe(true);
    } finally {
      if (result != null) tree.dispose(result);
      tree.dispose(init);
    }
  });

  it("solver pattern: scan → foriLoop → foriLoop with grad+hessian+candidates", () => {
    // Full solver pattern with multiple candidate evaluations
    using alpha = np.array(2.5);
    using beta = np.array(0.3);
    using gamma = np.array(1.7);

    const init = np.array(1.0);
    using xs = np.array([0.0]);

    let result: np.Array | null = null;
    let ys: np.Array | null = null;

    try {
      const scanBody = (carry: np.Array, _x: np.Array): [np.Array, np.Array] => {
        const hourlyBody = (i: np.Array, c: np.Array): np.Array => {
          const objFn = (params: np.Array): np.Array => {
            const a = lax.dynamicIndexInDim(params, 0, 0, false);
            const b = lax.dynamicIndexInDim(params, 1, 0, false);
            return a.mul(alpha).add(b.mul(beta)).sub(gamma);
          };

          type SolverState = { x: np.Array; lam: np.Array };
          const newtonBody = (_j: np.Array, state: SolverState): SolverState => {
            const fCur = objFn(state.x);
            const g = grad(objFn)(state.x);
            const h = hessian(objFn)(state.x);
            const eye = np.eye(2, { dtype: state.x.dtype });
            const step = g.mul(0.01);

            // Multiple candidates like the real solver
            const c1 = state.x.sub(step.mul(0.1));
            const c2 = state.x.sub(step.mul(0.3));
            const c3 = state.x.sub(step.mul(1.0));
            
            // Evaluate each candidate
            const v1 = objFn(c1);
            const v2 = objFn(c2);
            const v3 = objFn(c3);
            
            // Pick best
            const best12 = np.where(v1.less(v2), c1, c2);
            const bestV12 = np.where(v1.less(v2), v1, v2);
            const bestX = np.where(bestV12.less(v3), best12, c3);
            const bestV = np.where(bestV12.less(v3), bestV12, v3);
            
            const improved = bestV.less(fCur);
            const xNext = np.where(improved, bestX, state.x);
            
            return { x: xNext, lam: state.lam };
          };

          const solverInit: SolverState = {
            x: np.array([1.0, 0.5]),
            lam: np.array(0.01),
          };

          const solved = lax.foriLoop(
            0, 3,
            newtonBody as unknown as (i: np.Array, c: JsTree<np.Array>) => JsTree<np.Array>,
            solverInit as unknown as JsTree<np.Array>,
          ) as unknown as SolverState;

          return c.add(lax.dynamicIndexInDim(solved.x, 0, 0, false).mul(0.01));
        };

        const inner = lax.foriLoop(0, 2, hourlyBody, carry);
        return [inner, inner];
      };

      [result as unknown, ys] = lax.scan(
        scanBody as unknown as (c: JsTree<np.Array>, x: JsTree<np.Array>) => [JsTree<np.Array>, JsTree<np.Array>],
        init as unknown as JsTree<np.Array>,
        xs as unknown as JsTree<np.Array>,
        { length: 1 },
      );

      expect(Number.isFinite((result as np.Array).js() as number)).toBe(true);
    } finally {
      if (ys != null) (ys as np.Array).dispose();
      if (result != null) (result as np.Array).dispose();
      init.dispose();
    }
  });

  it("actual phydro solver in scan → foriLoop", () => {
    // Use the actual pmodelHydraulicsNumerical inside scan → foriLoop
    using tc = np.array(10.0);
    using ppfd = np.array(100.0);
    using vpd = np.array(500.0);
    using co2 = np.array(400.0);
    using pres = np.array(101325.0);
    using fapar = np.array(0.8);
    using psiSoil = np.array(-0.5);
    using rdark = np.array(0.02);
    using conductivity = np.array(3.0e-17);
    using psi50 = np.array(-3.46);
    using bParam = np.array(6.0);
    using alphaCost = np.array(0.1);
    using gammaCost = np.array(1.0);
    const KPHIO = 0.087182;

    const init = np.array(0.0);
    using xs = np.array([0.0]);

    let result: np.Array | null = null;
    let ys: np.Array | null = null;

    try {
      const scanBody = (carry: np.Array, _x: np.Array): [np.Array, np.Array] => {
        const hourlyBody = (_i: np.Array, c: np.Array): np.Array => {
          const phydro = pmodelHydraulicsNumerical(
            tc, ppfd, vpd, co2, pres, fapar,
            psiSoil,
            rdark, conductivity, psi50, bParam, alphaCost, gammaCost,
            KPHIO, "projected_newton",
          );
          using aj = phydro.aj.ref;
          tree.dispose(phydro);
          return c.add(aj);
        };

        const inner = lax.foriLoop(0, 1, hourlyBody, carry);
        return [inner, inner];
      };

      [result as unknown, ys] = lax.scan(
        scanBody as unknown as (c: JsTree<np.Array>, x: JsTree<np.Array>) => [JsTree<np.Array>, JsTree<np.Array>],
        init as unknown as JsTree<np.Array>,
        xs as unknown as JsTree<np.Array>,
        { length: 1 },
      );

      expect(Number.isFinite((result as np.Array).js() as number)).toBe(true);
    } finally {
      if (ys != null) (ys as np.Array).dispose();
      if (result != null) (result as np.Array).dispose();
      init.dispose();
    }
  });

  it("actual phydro solver in scan only (no foriLoop)", () => {
    using tc = np.array(10.0);
    using ppfd = np.array(100.0);
    using vpd = np.array(500.0);
    using co2 = np.array(400.0);
    using pres = np.array(101325.0);
    using fapar = np.array(0.8);
    using psiSoil = np.array(-0.5);
    using rdark = np.array(0.02);
    using conductivity = np.array(3.0e-17);
    using psi50 = np.array(-3.46);
    using bParam = np.array(6.0);
    using alphaCost = np.array(0.1);
    using gammaCost = np.array(1.0);
    const KPHIO = 0.087182;

    const init = np.array(0.0);
    using xs = np.array([0.0]);

    let result: np.Array | null = null;
    let ys: np.Array | null = null;

    try {
      const scanBody = (carry: np.Array, _x: np.Array): [np.Array, np.Array] => {
        const phydro = pmodelHydraulicsNumerical(
          tc, ppfd, vpd, co2, pres, fapar,
          psiSoil,
          rdark, conductivity, psi50, bParam, alphaCost, gammaCost,
          KPHIO, "projected_newton",
        );
        using aj = phydro.aj.ref;
        tree.dispose(phydro);
        const out = carry.add(aj);
        return [out, out];
      };

      [result as unknown, ys] = lax.scan(
        scanBody as unknown as (c: JsTree<np.Array>, x: JsTree<np.Array>) => [JsTree<np.Array>, JsTree<np.Array>],
        init as unknown as JsTree<np.Array>,
        xs as unknown as JsTree<np.Array>,
        { length: 1 },
      );

      expect(Number.isFinite((result as np.Array).js() as number)).toBe(true);
    } finally {
      if (ys != null) (ys as np.Array).dispose();
      if (result != null) (result as np.Array).dispose();
      init.dispose();
    }
  });

  it("actual phydro solver in foriLoop only (no scan)", () => {
    using tc = np.array(10.0);
    using ppfd = np.array(100.0);
    using vpd = np.array(500.0);
    using co2 = np.array(400.0);
    using pres = np.array(101325.0);
    using fapar = np.array(0.8);
    using psiSoil = np.array(-0.5);
    using rdark = np.array(0.02);
    using conductivity = np.array(3.0e-17);
    using psi50 = np.array(-3.46);
    using bParam = np.array(6.0);
    using alphaCost = np.array(0.1);
    using gammaCost = np.array(1.0);
    const KPHIO = 0.087182;

    const init = np.array(0.0);
    let result: np.Array | null = null;

    try {
      const hourlyBody = (_i: np.Array, c: np.Array): np.Array => {
        const phydro = pmodelHydraulicsNumerical(
          tc, ppfd, vpd, co2, pres, fapar,
          psiSoil,
          rdark, conductivity, psi50, bParam, alphaCost, gammaCost,
          KPHIO, "projected_newton",
        );
        using aj = phydro.aj.ref;
        tree.dispose(phydro);
        return c.add(aj);
      };

      result = lax.foriLoop(0, 1, hourlyBody, init) as np.Array;
      expect(Number.isFinite((result as np.Array).js() as number)).toBe(true);
    } finally {
      if (result != null) (result as np.Array).dispose();
      init.dispose();
    }
  });
});
