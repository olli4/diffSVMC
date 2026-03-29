/**
 * Daily biomass allocation (alloc_hypothesis_2).
 *
 * Fortran reference: vendor/SVMC/src/allocation.f90 L141–280
 *
 * Allocates daily GPP into leaf, stem, root, and grain carbon pools with
 * Q10-dependent maintenance and growth respiration, turnover-based litter
 * production, and management actions (harvest, grazing, organic fertilizer).
 *
 * All branches use np.where for differentiability through the complete
 * allocation logic.
 */

import { np } from "../precision.js";

const SPD = 3600.0 * 24.0; // seconds per day
const EPS = 1e-30;         // denominator guard

/** Guard the undefined x/0 case with a tiny sign-preserving fallback. */
function safeDenom(x: np.Array): np.Array {
  using absX = np.abs(x);
  using tooSmall = absX.less(EPS);
  using signX = np.sign(x);
  using fallback = signX.mul(EPS);
  return np.where(tooSmall, fallback, x);
}

/**
 * Compute daily biomass allocation from GPP to carbon pools.
 *
 * All operations stay on traced np.Array values so the full computation is
 * differentiable via jax-js-nonconsuming autodiff (jit/grad/vjp).
 *
 * @param tempDay - Daily mean temperature (°C).
 * @param gppDay - Gross primary production (kg C m⁻² s⁻¹).
 * @param leafRdarkDay - Leaf dark respiration (kg C m⁻² s⁻¹).
 * @param croot - Root carbon (kg C m⁻²).
 * @param cleaf - Leaf carbon (kg C m⁻²).
 * @param cstem - Stem carbon (kg C m⁻²).
 * @param cgrain - Grain carbon (kg C m⁻²).
 * @param litterCleafIn - Input leaf litter carbon (used when invert_option != 0).
 * @param grainFill - Grain-filling flux (kg C m⁻² s⁻¹).
 * @param cratioResp - Maintenance respiration fraction.
 * @param cratioLeaf - Leaf allocation fraction.
 * @param cratioRoot - Root allocation fraction.
 * @param cratioBiomass - Carbon fraction of biomass.
 * @param turnoverCleaf - Leaf turnover rate at 20°C.
 * @param turnoverCroot - Root/stem turnover rate at 20°C.
 * @param sla - Specific leaf area (m² kg⁻¹).
 * @param q10 - Q10 temperature coefficient.
 * @param invertOption - Inversion mode (0=none, 1=cratio_leaf, 2=turnover).
 * @param managementType - Management type (0=none, 1=harvest, 3=grazing, 4=organic).
 * @param managementCInput - Carbon input rate from management (kg C m⁻² s⁻¹).
 * @param managementCOutput - Carbon output rate from management (kg C m⁻² s⁻¹).
 * @param pftIsOat - 1.0 for oat, 0.0 for grass.
 * @param phenoStage - Phenological stage (1=growth, 2=dormancy).
 * @returns Object with all pool states and fluxes. Caller must dispose all returned arrays.
 */
export function allocHypothesis2Fn(
  tempDay: np.Array,
  gppDay: np.Array,
  leafRdarkDay: np.Array,
  croot: np.Array,
  cleaf: np.Array,
  cstem: np.Array,
  cgrain: np.Array,
  litterCleafIn: np.Array,
  grainFill: np.Array,
  cratioResp: np.Array,
  cratioLeaf: np.Array,
  cratioRoot: np.Array,
  cratioBiomass: np.Array,
  turnoverCleaf: np.Array,
  turnoverCroot: np.Array,
  sla: np.Array,
  q10: np.Array,
  invertOption: np.Array,
  managementType: np.Array,
  managementCInput: np.Array,
  managementCOutput: np.Array,
  pftIsOat: np.Array,
  phenoStage: np.Array,
): {
  nppDay: np.Array;
  autoResp: np.Array;
  croot: np.Array;
  cleaf: np.Array;
  cstem: np.Array;
  cgrain: np.Array;
  litterCleaf: np.Array;
  litterCroot: np.Array;
  compost: np.Array;
  lai: np.Array;
  abovebiomass: np.Array;
  belowbiomass: np.Array;
  yield: np.Array;
  grainFill: np.Array;
  phenoStage: np.Array;
} {
  using isGrowth = phenoStage.equal(1);
  using invIs0 = invertOption.equal(0.0);

  // ---------- GROWTH PHASE ----------

  // Q10 temperature factor
  using _tDiff = tempDay.sub(20.0);
  using _tRatio = _tDiff.div(10.0);
  using q10f = np.power(q10, _tRatio);

  // Growth respirations
  using _gppLeaf = gppDay.mul(cratioLeaf);
  using _gppLeafMinusRdark = _gppLeaf.sub(leafRdarkDay);
  using _gppLeafMinusRdarkSpd = _gppLeafMinusRdark.mul(SPD);
  using _grRespLeafPre = np.maximum(_gppLeafMinusRdarkSpd, 0.0);
  using grRespLeaf = _grRespLeafPre.mul(0.11);

  using _oneMinusCLCR = cratioLeaf.add(cratioRoot);
  using _oneVal = np.array(1.0);
  using _stemFrac = _oneVal.sub(_oneMinusCLCR);
  using _gppStem = gppDay.mul(_stemFrac);
  using _cstemResp1b = cstem.mul(cratioResp);
  using _cstemResp = _cstemResp1b.mul(q10f);
  using _stemNet = _gppStem.sub(_cstemResp);
  using _stemNetSpd = _stemNet.mul(SPD);
  using _grRespStemPre = np.maximum(_stemNetSpd, 0.0);
  using grRespStem = _grRespStemPre.mul(0.11);

  using _gppRoot = gppDay.mul(cratioRoot);
  using _gppRootMinusGF = _gppRoot.sub(grainFill);
  using _crootResp1b = croot.mul(cratioResp);
  using _crootResp = _crootResp1b.mul(q10f);
  using _rootNet = _gppRootMinusGF.sub(_crootResp);
  using _rootNetSpd = _rootNet.mul(SPD);
  using _grRespRootPre = np.maximum(_rootNetSpd, 0.0);
  using grRespRoot = _grRespRootPre.mul(0.11);

  using _cg1 = cgrain.mul(0.1);
  using _cg2 = _cg1.mul(cratioResp);
  using _cgrainResp = _cg2.mul(q10f);
  using _grainNet = grainFill.sub(_cgrainResp);
  using _grainNetSpd = _grainNet.mul(SPD);
  using _grRespGrainPre = np.maximum(_grainNetSpd, 0.0);
  using _grRespGrainOat = _grRespGrainPre.mul(0.11);
  using _isOat = pftIsOat.greater(0.5);
  using grRespGrain = np.where(_isOat, _grRespGrainOat, 0.0);

  // NPP and autotrophic respiration
  using _crootPlusCstem = croot.add(cstem);
  using _maintSC1 = _crootPlusCstem.mul(cratioResp);
  using maintSC = _maintSC1.mul(q10f);
  using _maintG1 = cgrain.mul(0.1);
  using _maintG2 = _maintG1.mul(cratioResp);
  using maintG = _maintG2.mul(q10f);
  using _totalMaint1 = maintSC.add(maintG);
  using _totalMaint = _totalMaint1.add(leafRdarkDay);
  using _nppPre = gppDay.sub(_totalMaint);
  using nppG = _nppPre.mul(SPD);
  using _totalMaintSpd = _totalMaint.mul(SPD);
  using _grRespLS = grRespLeaf.add(grRespStem);
  using _grRespLSR = _grRespLS.add(grRespRoot);
  using _grRespTotal = _grRespLSR.add(grRespGrain);
  using autoRespG = _totalMaintSpd.add(_grRespTotal);

  // Turnover-based litter
  using _lcTurn1 = cleaf.mul(turnoverCleaf);
  using litterCleafTurnover = _lcTurn1.mul(q10f);
  using litterCleafG = np.where(invIs0, litterCleafTurnover, litterCleafIn);
  using _lcStem1 = cstem.mul(turnoverCroot);
  using litterCstemG = _lcStem1.mul(q10f);
  using _lcRoot1 = croot.mul(turnoverCroot);
  using litterCrootG = _lcRoot1.mul(q10f);
  using compostG = np.array(0.0);

  // Update pools (before management)
  using _cleafGpp1 = gppDay.mul(cratioLeaf);
  using _cleafGppTerm = _cleafGpp1.mul(SPD);
  using _cleafBase = cleaf.add(_cleafGppTerm);
  using _cleafSub1 = _cleafBase.sub(litterCleafG);
  using _rdarkSpd = leafRdarkDay.mul(SPD);
  using _cleafSub2 = _cleafSub1.sub(_rdarkSpd);
  using _cleafInv0 = _cleafSub2.sub(grRespLeaf);
  using cleafG = np.where(invIs0, _cleafInv0, cleaf);

  using _cstemGpp1 = gppDay.mul(_stemFrac);
  using _cstemGppTerm = _cstemGpp1.mul(SPD);
  using _cstemResp1 = cstem.mul(cratioResp);
  using _cstemResp2 = _cstemResp1.mul(q10f);
  using _cstemRespSpd = _cstemResp2.mul(SPD);
  using _cstem1 = cstem.add(_cstemGppTerm);
  using _cstem2 = _cstem1.sub(_cstemRespSpd);
  using _cstem3 = _cstem2.sub(litterCstemG);
  using cstemG = _cstem3.sub(grRespStem);

  using _crootGppTerm = _gppRootMinusGF.mul(SPD);
  using _crootResp1 = croot.mul(cratioResp);
  using _crootResp2 = _crootResp1.mul(q10f);
  using _crootRespSpd = _crootResp2.mul(SPD);
  using _croot1 = croot.add(_crootGppTerm);
  using _croot2 = _croot1.sub(litterCrootG);
  using _croot3 = _croot2.sub(_crootRespSpd);
  using _crootPre = _croot3.sub(grRespRoot);
  using crootG = np.maximum(_crootPre, 0.0);

  using _cgrainGFSpd = grainFill.mul(SPD);
  using _cg01 = cgrain.mul(0.1);
  using _cg02 = _cg01.mul(cratioResp);
  using _cg03 = _cg02.mul(q10f);
  using _cgrainRespSpd = _cg03.mul(SPD);
  using _cgrainOat1 = cgrain.add(_cgrainGFSpd);
  using _cgrainOat2 = _cgrainOat1.sub(_cgrainRespSpd);
  using _cgrainOat = _cgrainOat2.sub(grRespGrain);
  using cgrainG = np.where(_isOat, _cgrainOat, 0.0);

  // --- Management ---
  using isHarvest = managementType.equal(1);
  using isGrazing = managementType.equal(3);
  using isOrganic = managementType.equal(4);

  // Harvest grass: proportional deduction from cleaf (if inv==0) and cstem
  using hgDeduct = managementCOutput.mul(SPD);
  using _hgCleafCstemSum = cleafG.add(cstemG);
  using _hgSafeDenom1 = safeDenom(_hgCleafCstemSum);
  using _hgCleafMul = hgDeduct.mul(cleafG);
  using _hgCleafFrac = _hgCleafMul.div(_hgSafeDenom1);
  using _hgCleafReduced = cleafG.sub(_hgCleafFrac);
  using hgCleaf = np.where(invIs0, _hgCleafReduced, cleafG);
  // cstem uses possibly-modified cleaf in denominator (sequential effect)
  using _hgCleafCstem2 = hgCleaf.add(cstemG);
  using _hgSafeDenom2 = safeDenom(_hgCleafCstem2);
  using _hgCstemMul = hgDeduct.mul(cstemG);
  using _hgCstemFrac = _hgCstemMul.div(_hgSafeDenom2);
  using hgCstem = cstemG.sub(_hgCstemFrac);

  // Harvest oat: overwrite litter with updated pools, zero everything
  // (ho_litter_cleaf = cleafG, ho_litter_croot = crootG, ho_litter_cstem = cstemG)

  // Harvest combined (select by pft)
  using hCleaf = np.where(_isOat, 0.0, hgCleaf);
  using hCstem = np.where(_isOat, 0.0, hgCstem);
  using hCroot = np.where(_isOat, 0.0, crootG);
  using hCgrain = np.where(_isOat, 0.0, cgrainG);
  using hNpp = np.where(_isOat, 0.0, nppG);
  using hResp = np.where(_isOat, 0.0, autoRespG);
  using hLitterCleaf = np.where(_isOat, cleafG, litterCleafG);
  using hLitterCroot = np.where(_isOat, crootG, litterCrootG);
  using hLitterCstem = np.where(_isOat, cstemG, litterCstemG);
  using hCompost = compostG;

  // Grazing: deduct from cleaf (if inv==0) and cstem, add compost
  using gzDeduct = managementCOutput.mul(SPD);
  using _gzCleafCstemSum = cleafG.add(cstemG);
  using _gzSafeDenom1 = safeDenom(_gzCleafCstemSum);
  using _gzCleafMul = gzDeduct.mul(cleafG);
  using _gzCleafFrac = _gzCleafMul.div(_gzSafeDenom1);
  using _gzCleafReduced = cleafG.sub(_gzCleafFrac);
  using gzCleaf = np.where(invIs0, _gzCleafReduced, cleafG);
  using _gzCleafCstem2 = gzCleaf.add(cstemG);
  using _gzSafeDenom2 = safeDenom(_gzCleafCstem2);
  using _gzCstemMul = gzDeduct.mul(cstemG);
  using _gzCstemFrac = _gzCstemMul.div(_gzSafeDenom2);
  using gzCstem = cstemG.sub(_gzCstemFrac);
  using gzCompost = managementCInput.mul(SPD);

  // Organic: only compost
  using orgCompost = managementCInput.mul(SPD);

  // Select by management type
  function sel(h: np.Array, gz: np.Array, org: np.Array, nm: np.Array): np.Array {
    using _inner1 = np.where(isOrganic, org, nm);
    using _inner2 = np.where(isGrazing, gz, _inner1);
    return np.where(isHarvest, h, _inner2);
  }

  using mCleaf = sel(hCleaf, gzCleaf, cleafG, cleafG);
  using mCstem = sel(hCstem, gzCstem, cstemG, cstemG);
  using mCroot = sel(hCroot, crootG, crootG, crootG);
  using mCgrain = sel(hCgrain, cgrainG, cgrainG, cgrainG);
  using mNpp = sel(hNpp, nppG, nppG, nppG);
  using mResp = sel(hResp, autoRespG, autoRespG, autoRespG);
  using mLitterCleaf = sel(hLitterCleaf, litterCleafG, litterCleafG, litterCleafG);
  using mLitterCroot = sel(hLitterCroot, litterCrootG, litterCrootG, litterCrootG);
  using mLitterCstem = sel(hLitterCstem, litterCstemG, litterCstemG, litterCstemG);
  using mCompost = sel(hCompost, gzCompost, orgCompost, compostG);

  // Combine leaf + stem litter
  using gLitterCleaf = mLitterCleaf.add(mLitterCstem);
  using gLitterCroot = mLitterCroot;

  // Derived quantities (growth phase)
  using _aboveLS = mCleaf.add(mCstem);
  using _gAboveSum = _aboveLS.add(mCgrain);
  using gAbove = _gAboveSum.div(cratioBiomass);
  using gBelow = mCroot.div(cratioBiomass);
  using _gLaiPre = mCleaf.div(cratioBiomass);
  using gLai = _gLaiPre.mul(sla);

  // ---------- DORMANCY PHASE (pheno_stage == 2) ----------
  // Original input values (NOT combined with cstem)
  using dLitterCleaf = cleaf;
  using dLitterCroot = croot;
  using dAbove = cgrain.div(cratioBiomass);
  using dBelow = np.array(0.0);
  using dLai = np.array(0.0);

  // ---------- SELECT BY PHENOLOGICAL STAGE ----------
  return {
    nppDay: np.where(isGrowth, mNpp, 0.0),
    autoResp: np.where(isGrowth, mResp, 0.0),
    croot: np.where(isGrowth, mCroot, 0.0),
    cleaf: np.where(isGrowth, mCleaf, 0.0),
    cstem: np.where(isGrowth, mCstem, 0.0),
    cgrain: np.where(isGrowth, mCgrain, cgrain),
    litterCleaf: np.where(isGrowth, gLitterCleaf, dLitterCleaf),
    litterCroot: np.where(isGrowth, gLitterCroot, dLitterCroot),
    compost: np.where(isGrowth, mCompost, 0.0),
    lai: np.where(isGrowth, gLai, dLai),
    abovebiomass: np.where(isGrowth, gAbove, dAbove),
    belowbiomass: np.where(isGrowth, gBelow, dBelow),
    yield: np.array(0.0),    // never modified
    grainFill,               // never modified — caller owns lifetime
    phenoStage: np.where(isGrowth, 1, 1),  // both paths reset to 1
  };
}
