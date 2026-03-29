import { describe, it, expect } from "vitest";
import { numpy as baseNp } from "@hamk-uas/jax-js-nonconsuming";
import { getNumericDType, np } from "../src/precision.js";
import { allocHypothesis2Fn, invertAllocFn } from "../src/allocation/index.js";
import allocFixtures from "../../svmc-ref/fixtures/allocation.json";

// Machine epsilon for the configured numeric dtype.
const eps = baseNp.finfo(getNumericDType()).eps;

// Allocation critical path: Q10 power (~5 ops), growth resp maxima (~3 ops),
// pool update (~10 ops), management select (~8 ops), derived quantities (~3 ops).
// Total ~30 ops. Conservative 4096 * eps.
const RTOL = 4096 * eps;

function pftFlag(pftType: string): number {
  return pftType === "oat" ? 1.0 : 0.0;
}

// ──────────────────────────────────────────────────────────────────
// alloc_hypothesis_2 — fixture playback
// ──────────────────────────────────────────────────────────────────

type AllocCase = (typeof allocFixtures.alloc_hypothesis_2)[number];

function allocLabel(c: AllocCase): string {
  const inp = c.inputs;
  return `pheno=${inp.pheno_stage}_inv=${inp.invert_option}_mgmt=${inp.management_type}_pft=${inp.pft_type}`;
}

describe("alloc_hypothesis_2 — Fortran reference", () => {
  for (const c of allocFixtures.alloc_hypothesis_2) {
    it(allocLabel(c), async () => {
      const inp = c.inputs;
      const out = c.output;
      const pft = pftFlag(inp.pft_type);

      using tempDay = np.array(inp.temp_day);
      using gppDay = np.array(inp.gpp_day);
      using leafRdarkDay = np.array(inp.leaf_rdark_day);
      using croot = np.array(inp.croot);
      using cleaf = np.array(inp.cleaf);
      using cstem = np.array(inp.cstem);
      using cgrain = np.array(inp.cgrain);
      using litterCleafIn = np.array(inp.litter_cleaf);
      using grainFill = np.array(inp.grain_fill);
      using cratioResp = np.array(inp.cratio_resp);
      using cratioLeaf = np.array(inp.cratio_leaf);
      using cratioRoot = np.array(inp.cratio_root);
      using cratioBiomass = np.array(inp.cratio_biomass);
      using turnoverCleaf = np.array(inp.turnover_cleaf);
      using turnoverCroot = np.array(inp.turnover_croot);
      using sla = np.array(inp.sla);
      using q10 = np.array(inp.q10);
      using invertOption = np.array(inp.invert_option);
      using managementType = np.array(inp.management_type);
      using managementCInput = np.array(inp.management_c_input);
      using managementCOutput = np.array(inp.management_c_output);
      using pftIsOat = np.array(pft);
      using phenoStage = np.array(inp.pheno_stage);

      const result = allocHypothesis2Fn(
        tempDay, gppDay, leafRdarkDay,
        croot, cleaf, cstem, cgrain,
        litterCleafIn, grainFill,
        cratioResp, cratioLeaf, cratioRoot, cratioBiomass,
        turnoverCleaf, turnoverCroot, sla, q10, invertOption,
        managementType, managementCInput, managementCOutput,
        pftIsOat, phenoStage,
      );

      // Dispose all returned arrays
      using _nppDay = result.nppDay;
      using _autoResp = result.autoResp;
      using _croot = result.croot;
      using _cleaf = result.cleaf;
      using _cstem = result.cstem;
      using _cgrain = result.cgrain;
      using _litterCleaf = result.litterCleaf;
      using _litterCroot = result.litterCroot;
      using _compost = result.compost;
      using _lai = result.lai;
      using _abovebiomass = result.abovebiomass;
      using _belowbiomass = result.belowbiomass;
      using _yield = result.yield;
      using _grainFill = result.grainFill;
      using _phenoStage = result.phenoStage;

      // Map from fixture key names to result property names
      const keyMap: Record<string, keyof typeof result> = {
        npp_day: "nppDay",
        auto_resp: "autoResp",
        croot: "croot",
        cleaf: "cleaf",
        cstem: "cstem",
        cgrain: "cgrain",
        litter_cleaf: "litterCleaf",
        litter_croot: "litterCroot",
        compost: "compost",
        lai: "lai",
        abovebiomass: "abovebiomass",
        belowbiomass: "belowbiomass",
        yield: "yield",
        grain_fill: "grainFill",
        pheno_stage: "phenoStage",
      };

      for (const [fixtureKey, value] of Object.entries(out)) {
        const resultKey = keyMap[fixtureKey];
        if (resultKey == null) continue;
        const actual = result[resultKey] as np.Array;
        using expected = np.array(value as number);
        expect(actual).toBeAllclose(expected, { rtol: RTOL });
      }
    });
  }
});

// ──────────────────────────────────────────────────────────────────
// invert_alloc — fixture playback
// ──────────────────────────────────────────────────────────────────

type InvertCase = (typeof allocFixtures.invert_alloc)[number];

function invertLabel(c: InvertCase): string {
  const inp = c.inputs;
  return `pheno=${inp.pheno_stage}_inv=${inp.invert_option}_mgmt=${inp.management_type}_pft=${inp.pft_type}`;
}

describe("invert_alloc — Fortran reference", () => {
  for (const c of allocFixtures.invert_alloc) {
    it(invertLabel(c), async () => {
      const inp = c.inputs;
      const out = c.output;
      const pft = pftFlag(inp.pft_type);

      using deltaLai = np.array(inp.delta_lai);
      using leafRdarkDay = np.array(inp.leaf_rdark_day);
      using tempDay = np.array(inp.temp_day);
      using gppDay = np.array(inp.gpp_day);
      using litterCleafIn = np.array(inp.litter_cleaf);
      using cleaf = np.array(inp.cleaf);
      using cstem = np.array(inp.cstem);
      using cratioResp = np.array(inp.cratio_resp);
      using cratioLeaf = np.array(inp.cratio_leaf);
      using cratioRoot = np.array(inp.cratio_root);
      using cratioBiomass = np.array(inp.cratio_biomass);
      using turnoverCleaf = np.array(inp.turnover_cleaf);
      using sla = np.array(inp.sla);
      using q10 = np.array(inp.q10);
      using invertOption = np.array(inp.invert_option);
      using managementType = np.array(inp.management_type);
      using managementCOutput = np.array(inp.management_c_output);
      using pftIsOat = np.array(pft);
      using phenoStage = np.array(inp.pheno_stage);

      const result = invertAllocFn(
        deltaLai, leafRdarkDay, tempDay, gppDay,
        litterCleafIn, cleaf, cstem,
        cratioResp, cratioLeaf, cratioRoot, cratioBiomass,
        turnoverCleaf, sla, q10, invertOption,
        managementType, managementCOutput,
        pftIsOat, phenoStage,
      );

      // Dispose all returned arrays
      using _deltaLai = result.deltaLai;
      using _litterCleaf = result.litterCleaf;
      using _cleaf = result.cleaf;
      using _cratioLeaf = result.cratioLeaf;
      using _cratioRoot = result.cratioRoot;
      using _turnoverCleaf = result.turnoverCleaf;

      const keyMap: Record<string, keyof typeof result> = {
        delta_lai: "deltaLai",
        litter_cleaf: "litterCleaf",
        cleaf: "cleaf",
        cratio_leaf: "cratioLeaf",
        cratio_root: "cratioRoot",
        turnover_cleaf: "turnoverCleaf",
      };

      for (const [fixtureKey, value] of Object.entries(out)) {
        const resultKey = keyMap[fixtureKey];
        if (resultKey == null) continue;
        const actual = result[resultKey] as np.Array;
        using expected = np.array(value as number);
        expect(actual).toBeAllclose(expected, { rtol: RTOL });
      }
    });
  }
});
