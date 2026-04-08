/**
 * LAI-to-leaf-carbon inversion (invert_alloc).
 *
 * Fortran reference: vendor/SVMC/src/allocation.f90 L296–420
 *
 * Derives allometric parameters (cratio_leaf or turnover_cleaf) from
 * observed LAI changes.  Mode is controlled by invert_option:
 *   1 → derive cratio_leaf from delta_lai
 *   2 → derive turnover_cleaf from delta_lai
 *
 * All branches use np.where for differentiability through the complete
 * inversion logic.
 */

import { np } from "../precision.js";

const SPD = 3600.0 * 24.0; // seconds per day
const EPS = 1e-30;         // denominator guard

/** Guard the undefined x/0 case with a tiny sign-preserving fallback. */
function safeDenom(x: np.Array): np.Array {
  using absX = np.abs(x);
  using tooSmall = absX.less(EPS);
  using signX = np.sign(x);
  using zeroSign = signX.equal(0.0);
  using fallbackSign = np.where(zeroSign, 1.0, signX);
  using fallback = fallbackSign.mul(EPS);
  return np.where(tooSmall, fallback, x);
}

/**
 * Derive allometric parameters from LAI change.
 *
 * All operations stay on traced np.Array values so the full computation is
 * differentiable via jax-js-nonconsuming autodiff (jit/grad/vjp).
 *
 * @param deltaLai - Observed change in LAI.
 * @param leafRdarkDay - Leaf dark respiration (kg C m⁻² s⁻¹).
 * @param tempDay - Daily mean temperature (°C).
 * @param gppDay - Gross primary production (kg C m⁻² s⁻¹).
 * @param litterCleafIn - Input leaf litter carbon.
 * @param cleaf - Leaf carbon (kg C m⁻²).
 * @param cstem - Stem carbon (kg C m⁻²).
 * @param cratioResp - Maintenance respiration fraction.
 * @param cratioLeaf - Leaf allocation fraction (input; derived as output for option 1).
 * @param cratioRoot - Root allocation fraction (input; derived as output for option 1).
 * @param cratioBiomass - Carbon fraction of biomass.
 * @param harvestIndex - Retained for fixture/JAX signature parity; not used here.
 * @param turnoverCleaf - Leaf turnover rate (input; derived as output for option 2).
 * @param turnoverCroot - Retained for fixture/JAX signature parity; not used here.
 * @param sla - Specific leaf area (m² kg⁻¹).
 * @param q10 - Q10 temperature coefficient.
 * @param invertOption - Inversion mode (1=derive cratio_leaf, 2=derive turnover).
 * @param managementType - 0=none, 1=harvest, 3=grazing, 4=organic.
 * @param managementCInput - Retained for fixture/JAX signature parity; not used here.
 * @param managementCOutput - Carbon output rate (kg C m⁻² s⁻¹).
 * @param pftIsOat - 1.0 for oat, 0.0 for grass.
 * @param phenoStage - 1=growth (active), 2=dormancy (no-op).
 * @returns Object with keys: deltaLai, litterCleaf, cleaf, cratioLeaf,
 *   cratioRoot, turnoverCleaf. Caller must dispose all returned arrays.
 */
export function invertAllocFn(
  deltaLai: np.Array,
  leafRdarkDay: np.Array,
  tempDay: np.Array,
  gppDay: np.Array,
  litterCleafIn: np.Array,
  cleaf: np.Array,
  cstem: np.Array,
  cratioResp: np.Array,
  cratioLeaf: np.Array,
  cratioRoot: np.Array,
  cratioBiomass: np.Array,
  _harvestIndex: np.Array,
  turnoverCleaf: np.Array,
  _turnoverCroot: np.Array,
  sla: np.Array,
  q10: np.Array,
  invertOption: np.Array,
  managementType: np.Array,
  _managementCInput: np.Array,
  managementCOutput: np.Array,
  pftIsOat: np.Array,
  phenoStage: np.Array,
): {
  deltaLai: np.Array;
  litterCleaf: np.Array;
  cleaf: np.Array;
  cratioLeaf: np.Array;
  cratioRoot: np.Array;
  turnoverCleaf: np.Array;
} {
  using isActive = phenoStage.equal(1);

  // Common intermediates
  using _tDiff = tempDay.sub(20.0);
  using _tRatio = _tDiff.div(10.0);
  using q10f = np.power(q10, _tRatio);

  using _dLaiOverSla = deltaLai.div(sla);
  using deltaCleaf = _dLaiOverSla.mul(cratioBiomass);

  using _gppLeaf = gppDay.mul(cratioLeaf);
  using _gppLeafMinusRdark = _gppLeaf.sub(leafRdarkDay);
  using _gppLeafMinusRdarkSpd = _gppLeafMinusRdark.mul(SPD);
  using _grRespLeafPre = np.maximum(_gppLeafMinusRdarkSpd, 0.0);
  using grRespLeaf = _grRespLeafPre.mul(0.11);

  // Management output term (for grass pft in harvest/grazing)
  using _mgmtOutSpd = managementCOutput.mul(SPD);
  using _cleafPlusCstem = cleaf.add(cstem);
  using _safeDenom1 = safeDenom(_cleafPlusCstem);
  using _mgmtFrac = cleaf.div(_safeDenom1);
  using mgmtOutTerm = _mgmtOutSpd.mul(_mgmtFrac);

  // Only grass harvest and grass grazing add the management term
  using isHarvest = managementType.equal(1);
  using isGrazing = managementType.equal(3);
  using isGrass = pftIsOat.less(0.5);

  using _hvGrassCond = isHarvest.mul(isGrass);
  using mgmtTermHarvest = np.where(_hvGrassCond, mgmtOutTerm, 0.0);
  using _gzGrassCond = isGrazing.mul(isGrass);
  using mgmtTermGrazing = np.where(_gzGrassCond, mgmtOutTerm, 0.0);
  using _mgmtInner = np.where(isGrazing, mgmtTermGrazing, 0.0);
  using mgmtTerm = np.where(isHarvest, mgmtTermHarvest, _mgmtInner);

  // ===== OPTION 1: derive cratio_leaf =====
  using _lc1_1 = cleaf.mul(turnoverCleaf);
  using litterCleaf1 = _lc1_1.mul(q10f);
  using gppAboveThreshold = gppDay.greater(0.2e-8);

  using _rawNum = deltaCleaf.add(litterCleaf1);
  using _rawNum2a = leafRdarkDay.mul(SPD);
  using _rawNum2 = _rawNum.add(_rawNum2a);
  using _rawNum3 = _rawNum2.add(grRespLeaf);
  using _rawNum4 = _rawNum3.add(mgmtTerm);
  using _ratioNum = np.maximum(gppDay, EPS);
  using _rawDenomSpd = _ratioNum.mul(SPD);
  using _rawRatio = _rawNum4.div(_rawDenomSpd);
  using rawCratio = np.clip(_rawRatio, 0.1, 0.9);

  using cratioLeaf1 = np.where(gppAboveThreshold, rawCratio, cratioLeaf);
  using _negCL1 = cratioLeaf1.neg();
  using _oneMinusCL1 = _negCL1.add(1.0);
  using cratioRoot1 = np.where(gppAboveThreshold, _oneMinusCL1, cratioRoot);
  using _cleafPlusDelta = cleaf.add(deltaCleaf);
  using cleaf1 = np.maximum(_cleafPlusDelta, 0.0);

  // ===== OPTION 2: derive turnover_cleaf =====
  using cleafAboveThreshold = cleaf.greater(0.00001);

  using _gppCL = gppDay.mul(cratioLeaf);
  using _gppCLSpd = _gppCL.mul(SPD);
  using _rawTNum = _gppCLSpd.sub(deltaCleaf);
  using _rawTNum2a = leafRdarkDay.mul(SPD);
  using _rawTNum2 = _rawTNum.sub(_rawTNum2a);
  using _rawTNum3 = _rawTNum2.sub(grRespLeaf);
  using _rawTNum4 = _rawTNum3.sub(mgmtTerm);
  using _safeCleaf = np.maximum(cleaf, EPS);
  using _safeQ10f = np.maximum(q10f, EPS);
  using _rawTDiv1 = _rawTNum4.div(_safeCleaf);
  using _rawTurnover = _rawTDiv1.div(_safeQ10f);
  using turnoverCleaf2 = np.where(cleafAboveThreshold, _rawTurnover, turnoverCleaf);

  using cleaf2pre = np.where(cleafAboveThreshold, cleaf, 0.0);
  using _lc2_1 = cleaf2pre.mul(turnoverCleaf2);
  using litterCleaf2 = _lc2_1.mul(q10f);
  using _cleaf2prePlusDelta = cleaf2pre.add(deltaCleaf);
  using cleaf2 = np.maximum(_cleaf2prePlusDelta, 0.0);

  // ===== SELECT BY INVERT OPTION =====
  using isOpt1 = invertOption.equal(1.0);
  using isOpt2 = invertOption.equal(2.0);

  using _lcInner = np.where(isOpt2, litterCleaf2, litterCleafIn);
  using outLitterCleaf = np.where(isOpt1, litterCleaf1, _lcInner);
  using _clInner = np.where(isOpt2, cleaf2, cleaf);
  using outCleaf = np.where(isOpt1, cleaf1, _clInner);
  using outCratioLeaf = np.where(isOpt1, cratioLeaf1, cratioLeaf);
  using outCratioRoot = np.where(isOpt1, cratioRoot1, cratioRoot);
  using outTurnover = np.where(isOpt2, turnoverCleaf2, turnoverCleaf);

  // ===== INACTIVE (pheno_stage != 1): pass through =====
  return {
    deltaLai: deltaLai.ref,
    litterCleaf: np.where(isActive, outLitterCleaf, litterCleafIn),
    cleaf: np.where(isActive, outCleaf, cleaf),
    cratioLeaf: np.where(isActive, outCratioLeaf, cratioLeaf),
    cratioRoot: np.where(isActive, outCratioRoot, cratioRoot),
    turnoverCleaf: np.where(isActive, outTurnover, turnoverCleaf),
  };
}
