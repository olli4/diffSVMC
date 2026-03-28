"""LAI-to-leaf-carbon inversion (invert_alloc).

Fortran reference: vendor/SVMC/src/allocation.f90 L296–420

Derives allometric parameters (cratio_leaf or turnover_cleaf) from
observed LAI changes.  Mode is controlled by invert_option:
  1 → derive cratio_leaf from delta_lai
  2 → derive turnover_cleaf from delta_lai

The implementation uses jnp.where for all conditional branches to
maintain JAX differentiability.
"""

import jax.numpy as jnp

_SPD = 3600.0 * 24.0  # seconds per day


def invert_alloc(
    delta_lai, leaf_rdark_day, temp_day, gpp_day,
    litter_cleaf_in, cleaf, cstem,
    cratio_resp, cratio_leaf, cratio_root, cratio_biomass,
    harvest_index, turnover_cleaf, turnover_croot, sla, q10, invert_option,
    management_type, management_c_input, management_c_output,
    pft_is_oat, pheno_stage,
):
    """Derive allometric parameters from LAI change.

    Fortran reference: subroutine invert_alloc in allocation.f90

    Args:
        delta_lai: Observed change in LAI.
        leaf_rdark_day: Leaf dark respiration (kg C m⁻² s⁻¹).
        temp_day: Daily mean temperature (°C).
        gpp_day: Gross primary production (kg C m⁻² s⁻¹).
        litter_cleaf_in: Input leaf litter carbon.
        cleaf: Leaf carbon (kg C m⁻²).
        cstem: Stem carbon (kg C m⁻²).
        cratio_resp: Maintenance respiration fraction.
        cratio_leaf: Leaf allocation fraction (input; derived as output for option 1).
        cratio_root: Root allocation fraction (input; derived as output for option 1).
        cratio_biomass: Carbon fraction of biomass.
        harvest_index: Harvest index (not used in invert_alloc).
        turnover_cleaf: Leaf turnover rate (input; derived as output for option 2).
        turnover_croot: Root turnover rate (not modified).
        sla: Specific leaf area (m² kg⁻¹).
        q10: Q10 temperature coefficient.
        invert_option: Inversion mode (1=derive cratio_leaf, 2=derive turnover).
        management_type: 0=none, 1=harvest, 3=grazing, 4=organic.
        management_c_input: Carbon input rate (kg C m⁻² s⁻¹).
        management_c_output: Carbon output rate (kg C m⁻² s⁻¹).
        pft_is_oat: 1.0 for oat, 0.0 for grass.
        pheno_stage: 1=growth (active), 2=dormancy (no-op).

    Returns:
        Dict with keys: delta_lai, litter_cleaf, cleaf,
        cratio_leaf, cratio_root, turnover_cleaf.
    """
    spd = _SPD
    is_active = pheno_stage == 1
    _eps = 1e-30  # denominator guard

    def _safe_denom(x):
        """Guard the undefined x/0 case with a tiny sign-preserving fallback."""
        return jnp.where(jnp.abs(x) < _eps, jnp.copysign(_eps, x), x)

    # Common intermediates (computed regardless of path, zero-cost in jnp.where)
    q10f = q10 ** ((temp_day - 20.0) / 10.0)
    delta_cleaf = delta_lai / sla * cratio_biomass
    gr_resp_leaf = jnp.maximum(
        0.0, (gpp_day * cratio_leaf - leaf_rdark_day) * spd) * 0.11

    # Management output term (for grass pft in harvest/grazing)
    mgmt_out_term = management_c_output * spd * cleaf / _safe_denom(cleaf + cstem)
    # Only grass harvest and grass grazing add the management term
    is_harvest = management_type == 1
    is_grazing = management_type == 3
    is_grass = 1.0 - pft_is_oat
    mgmt_term_harvest = jnp.where(is_harvest * is_grass, mgmt_out_term, 0.0)
    mgmt_term_grazing = jnp.where(is_grazing * is_grass, mgmt_out_term, 0.0)
    mgmt_term = jnp.where(is_harvest, mgmt_term_harvest,
                jnp.where(is_grazing, mgmt_term_grazing, 0.0))

    # ===== OPTION 1: derive cratio_leaf =====
    litter_cleaf_1 = cleaf * turnover_cleaf * q10f
    gpp_above_threshold = gpp_day > 0.2e-8
    raw_cratio = (delta_cleaf + litter_cleaf_1 + leaf_rdark_day * spd
                  + gr_resp_leaf + mgmt_term) / spd / jnp.maximum(gpp_day, _eps)
    cratio_leaf_1 = jnp.where(
        gpp_above_threshold,
        jnp.clip(raw_cratio, 0.1, 0.9),
        cratio_leaf)  # below threshold: unchanged
    cratio_root_1 = jnp.where(
        gpp_above_threshold,
        1.0 - cratio_leaf_1,
        cratio_root)  # below threshold: unchanged
    cleaf_1 = jnp.maximum(0.0, cleaf + delta_cleaf)

    # ===== OPTION 2: derive turnover_cleaf =====
    cleaf_above_threshold = cleaf > 0.00001
    raw_turnover = (
        (gpp_day * cratio_leaf * spd - delta_cleaf
         - leaf_rdark_day * spd - gr_resp_leaf - mgmt_term)
        / jnp.maximum(cleaf, _eps)
        / jnp.maximum(q10f, _eps)
    )
    turnover_cleaf_2 = jnp.where(cleaf_above_threshold, raw_turnover, turnover_cleaf)
    cleaf_2_pre = jnp.where(cleaf_above_threshold, cleaf, 0.0)
    litter_cleaf_2 = cleaf_2_pre * turnover_cleaf_2 * q10f
    cleaf_2 = jnp.maximum(0.0, cleaf_2_pre + delta_cleaf)

    # ===== SELECT BY INVERT OPTION =====
    is_opt1 = invert_option == 1.0
    is_opt2 = invert_option == 2.0

    out_litter_cleaf = jnp.where(is_opt1, litter_cleaf_1,
                       jnp.where(is_opt2, litter_cleaf_2, litter_cleaf_in))
    out_cleaf = jnp.where(is_opt1, cleaf_1,
                jnp.where(is_opt2, cleaf_2, cleaf))
    out_cratio_leaf = jnp.where(is_opt1, cratio_leaf_1, cratio_leaf)
    out_cratio_root = jnp.where(is_opt1, cratio_root_1, cratio_root)
    out_turnover = jnp.where(is_opt2, turnover_cleaf_2, turnover_cleaf)

    # ===== INACTIVE (pheno_stage != 1): pass through =====
    return {
        "delta_lai": delta_lai,  # never modified
        "litter_cleaf": jnp.where(is_active, out_litter_cleaf, litter_cleaf_in),
        "cleaf": jnp.where(is_active, out_cleaf, cleaf),
        "cratio_leaf": jnp.where(is_active, out_cratio_leaf, cratio_leaf),
        "cratio_root": jnp.where(is_active, out_cratio_root, cratio_root),
        "turnover_cleaf": jnp.where(is_active, out_turnover, turnover_cleaf),
    }
