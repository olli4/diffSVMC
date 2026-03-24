import { describe, it, expect } from "vitest";
import { inputsToFractions } from "../src/yasso/index.js";
import yassoFixtures from "../../svmc-ref/fixtures/yasso.json";

describe("Yasso module leaf functions — Fortran reference", () => {
  for (const c of yassoFixtures.inputs_to_fractions) {
    it(`inputs_to_fractions: leaf=${c.inputs.leaf}, root=${c.inputs.root}, sol=${c.inputs.soluble}, comp=${c.inputs.compost}`, async () => {
      using result = inputsToFractions(
        c.inputs.leaf,
        c.inputs.root,
        c.inputs.soluble,
        c.inputs.compost,
      );
      expect(result).toBeAllclose(c.output as number[], { rtol: 5e-3 });
    });
  }
});
