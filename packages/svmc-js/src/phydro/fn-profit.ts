import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import { calcGs } from "./calc-gs.js";
import { calcAssimLightLimited } from "./calc-assim-light-limited.js";
import type {
  ParPlant,
  ParEnv,
  ParCost,
  ParPhotosynth,
} from "./scale-conductivity.js";

/**
 * Profit function for P-Hydro optimization.
 *
 * profit = Aj - α·Jmax - γ·Δψ²  (Profit Maximisation)
 *
 * Fortran: `fn_profit` in phydro_mod.f90
 *
 * @param logJmax - log(Jmax) optimization parameter
 * @param dpsi - Δψ optimization parameter (MPa)
 * @param psiSoil - Soil water potential (MPa)
 * @param parCost - Cost parameters
 * @param parPhotosynth - Photosynthesis parameters
 * @param parPlant - Plant hydraulic parameters
 * @param parEnv - Environmental parameters
 * @param hypothesis - "PM" (Profit Maximisation) or "LC" (Least Cost)
 * @param doOptim - When true, negate the result for minimizer-style optimization
 * @returns Profit value (µmol/m²/s)
 */
export function fnProfit(
  logJmax: np.Array,
  dpsi: np.Array,
  psiSoil: np.Array,
  parCost: ParCost,
  parPhotosynth: ParPhotosynth,
  parPlant: ParPlant,
  parEnv: ParEnv,
  hypothesis: "PM" | "LC" = "PM",
  doOptim = false,
): np.Array {
  using jmax = np.exp(logJmax);

  using gs = calcGs(dpsi, psiSoil, parPlant, parEnv);
  const { ci, aj: _aj } = calcAssimLightLimited(gs, jmax, parPhotosynth);
  ci.dispose();
  using aj = _aj;

  // costs = alpha * jmax + gamma * dpsi²
  using costJmax = parCost.alpha.mul(jmax);
  using _jaxTmp1 = dpsi.mul(dpsi);
  using costDpsi = parCost.gamma.mul(_jaxTmp1);
  using costs = costJmax.add(costDpsi);

  let profit: np.Array;
  if (hypothesis === "PM") {
    profit = aj.sub(costs);
  } else {
    using ajSafe = aj.add(1e-4);
    using negCosts = costs.mul(-1);
    profit = negCosts.div(ajSafe);
  }

  if (doOptim) {
    const negProfit = profit.mul(-1);
    profit.dispose();
    return negProfit;
  }

  return profit;
}
