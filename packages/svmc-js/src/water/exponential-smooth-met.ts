import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import waterConstants from "../../../svmc-ref/constants/water.json";

/**
 * Exponential smoothing for scaling meteorological parameters from daily to
 * monthly level.
 *
 * Fortran: `exponential_smooth_met` in wrapper_yasso.f90
 *
 * @param metDaily - Current daily meteorological values [2]
 * @param metRolling - Previous rolling average values [2]
 * @param metInd - Step counter (1 on first call)
 * @returns { metRolling, metInd } — updated rolling average and counter
 */
export function exponentialSmoothMet(
  metDaily: np.Array,
  metRolling: np.Array,
  metInd: number,
): { metRolling: np.Array; metInd: number } {
  const alphaSmooth1 = waterConstants.alpha_smooth1;
  const alphaSmooth2 = waterConstants.alpha_smooth2;

  if (metInd === 1) {
    // First call: initialize rolling to daily values
    const newRolling = metDaily.add(0); // clone
    return { metRolling: newRolling, metInd: metInd + 1 };
  }

  // Exponential smoothing: rolling = α·daily + (1−α)·rolling
  // .slice(i) extracts scalar element i from a 1-D array
  using d0 = metDaily.slice(0);
  using d1 = metDaily.slice(1);
  using r0 = metRolling.slice(0);
  using r1 = metRolling.slice(1);

  using _d0_scaled = d0.mul(alphaSmooth1);
  using _r0_scaled = r0.mul(1.0 - alphaSmooth1);
  using new0 = _d0_scaled.add(_r0_scaled);

  using _d1_scaled = d1.mul(alphaSmooth2);
  using _r1_scaled = r1.mul(1.0 - alphaSmooth2);
  using new1 = _d1_scaled.add(_r1_scaled);

  const newRolling = np.stack([new0, new1]);
  return { metRolling: newRolling, metInd };
}
