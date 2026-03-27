import { describe, it, expect } from "vitest";
import { matrixExp, matrixNorm } from "../src/yasso/index.js";
import { np } from "../src/precision.js";
import yassoFixtures from "../../svmc-ref/fixtures/yasso.json";

function scalingExponentForFixture(a: number[][]): number {
  const frobeniusNorm = Math.sqrt(
    a.flat().reduce((sum, value) => sum + value ** 2, 0),
  );
  return Math.max(1, Math.floor(Math.log2(Math.max(frobeniusNorm, 1))) + 1);
}

describe("matrixnorm — Fortran reference", () => {
  for (const c of yassoFixtures.matrixnorm) {
    const diag = (c.inputs.a as number[][]).map((r, i) => r[i]);
    it(`matrixnorm: diag=[${diag.map((v) => v.toFixed(2)).join(",")}]`, async () => {
      using a = np.array(c.inputs.a as number[][]);
      using result = matrixNorm(a);
      expect(result).toBeAllclose(c.output as number, { rtol: 1e-5 });
    });
  }
});

describe("matrixexp — Fortran reference", () => {
  for (const c of yassoFixtures.matrixexp) {
    const diag = (c.inputs.a as number[][]).map((r, i) => r[i]);
    it(`matrixexp: diag=[${diag.map((v) => v.toFixed(2)).join(",")}]`, async () => {
      using a = np.array(c.inputs.a as number[][]);
      using result = matrixExp(a);
      expect(result).toBeAllclose(c.output as number[][], { rtol: 1e-4 });
    });
  }
});

describe("matrixexp bounded squaring policy", () => {
  it("current reference cases stay within MAX_J=20", () => {
    const maxFixtureExponent = Math.max(
      ...yassoFixtures.matrixexp.map((c) =>
        scalingExponentForFixture(c.inputs.a as number[][]),
      ),
    );
    expect(maxFixtureExponent).toBeLessThanOrEqual(20);
  });
});
