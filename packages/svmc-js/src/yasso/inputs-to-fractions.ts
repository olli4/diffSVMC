import { np } from "../precision.js";
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
  leaf: number,
  root: number,
  soluble: number,
  compost: number,
): np.Array {
  using _leafArr = np.array(AWENH_LEAF);
  using _rootArr = np.array(AWENH_FINEROOT);
  using _solArr = np.array(AWENH_SOLUBLE);
  using _compArr = np.array(AWENH_COMPOST);

  using _l = _leafArr.mul(leaf);
  using _r = _rootArr.mul(root);
  using _s = _solArr.mul(soluble);
  using _c = _compArr.mul(compost);

  using _lr = _l.add(_r);
  using _lrs = _lr.add(_s);
  return _lrs.add(_c);
}
