import { retainArray, np } from "../precision.js";
import yassoConstants from "../../../svmc-ref/constants/yasso.json";

// AWENH fractions — canonical source: packages/svmc-ref/constants/yasso.json
const AWENH_LEAF = yassoConstants.AWENH_LEAF;
const AWENH_FINEROOT = yassoConstants.AWENH_FINEROOT;
const AWENH_SOLUBLE = yassoConstants.AWENH_SOLUBLE;
const AWENH_COMPOST = yassoConstants.AWENH_COMPOST;

/**
 * Split carbon inputs into AWENH pool fractions.
 *
 * Fortran: `inputs_to_fractions` in yasso.f90
 *
 * The fifth pool (H — Humus) never receives external input.
 *
 * @param leaf - Leaf litter carbon input
 * @param root - Fine-root litter carbon input
 * @param soluble - Soluble carbon input
 * @param compost - Compost carbon input
 * @returns Array of length 5 — AWENH fractions
 */
export function inputsToFractions(
  leaf: np.ArrayLike,
  root: np.ArrayLike,
  soluble: np.ArrayLike,
  compost: np.ArrayLike,
): np.Array {
  using leafArr = retainArray(leaf);
  using rootArr = retainArray(root);
  using solubleArr = retainArray(soluble);
  using compostArr = retainArray(compost);

  using leaf0 = leafArr.mul(AWENH_LEAF[0]);
  using root0 = rootArr.mul(AWENH_FINEROOT[0]);
  using soluble0 = solubleArr.mul(AWENH_SOLUBLE[0]);
  using compost0 = compostArr.mul(AWENH_COMPOST[0]);
  using out0a = leaf0.add(root0);
  using out0b = out0a.add(soluble0);
  using out0 = out0b.add(compost0);

  using leaf1 = leafArr.mul(AWENH_LEAF[1]);
  using root1 = rootArr.mul(AWENH_FINEROOT[1]);
  using soluble1 = solubleArr.mul(AWENH_SOLUBLE[1]);
  using compost1 = compostArr.mul(AWENH_COMPOST[1]);
  using out1a = leaf1.add(root1);
  using out1b = out1a.add(soluble1);
  using out1 = out1b.add(compost1);

  using leaf2 = leafArr.mul(AWENH_LEAF[2]);
  using root2 = rootArr.mul(AWENH_FINEROOT[2]);
  using soluble2 = solubleArr.mul(AWENH_SOLUBLE[2]);
  using compost2 = compostArr.mul(AWENH_COMPOST[2]);
  using out2a = leaf2.add(root2);
  using out2b = out2a.add(soluble2);
  using out2 = out2b.add(compost2);

  using leaf3 = leafArr.mul(AWENH_LEAF[3]);
  using root3 = rootArr.mul(AWENH_FINEROOT[3]);
  using soluble3 = solubleArr.mul(AWENH_SOLUBLE[3]);
  using compost3 = compostArr.mul(AWENH_COMPOST[3]);
  using out3a = leaf3.add(root3);
  using out3b = out3a.add(soluble3);
  using out3 = out3b.add(compost3);

  using leaf4 = leafArr.mul(AWENH_LEAF[4]);
  using root4 = rootArr.mul(AWENH_FINEROOT[4]);
  using soluble4 = solubleArr.mul(AWENH_SOLUBLE[4]);
  using compost4 = compostArr.mul(AWENH_COMPOST[4]);
  using out4a = leaf4.add(root4);
  using out4b = out4a.add(soluble4);
  using out4 = out4b.add(compost4);

  return np.stack([out0, out1, out2, out3, out4]);
}
