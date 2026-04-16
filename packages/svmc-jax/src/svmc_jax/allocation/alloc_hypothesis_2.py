"""Daily biomass allocation (alloc_hypothesis_2).

Fortran reference: vendor/SVMC/src/allocation.f90 L141–280

Allocates daily GPP into leaf, stem, root, and grain carbon pools with
Q10-dependent maintenance and growth respiration, turnover-based litter
production, and management actions (harvest, grazing, organic fertilizer).

The implementation uses jnp.where for all branches to maintain JAX
differentiability through the complete allocation logic.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

_SPD = 3600.0 * 24.0  # seconds per day


class AllocHypothesisResult(NamedTuple):
    """Typed allocation result container with backward-compatible key access."""

    npp_day: jnp.ndarray
    auto_resp: jnp.ndarray
    croot: jnp.ndarray
    cleaf: jnp.ndarray
    cstem: jnp.ndarray
    cgrain: jnp.ndarray
    litter_cleaf: jnp.ndarray
    litter_croot: jnp.ndarray
    compost: jnp.ndarray
    lai: jnp.ndarray
    abovebiomass: jnp.ndarray
    belowbiomass: jnp.ndarray
    yield_: jnp.ndarray
    grain_fill: jnp.ndarray
    pheno_stage: jnp.ndarray

    def __getitem__(self, item):
        if item == "yield":
            return self.yield_
        if isinstance(item, str):
            return getattr(self, item)
        return tuple.__getitem__(self, item)


@functools.partial(jax.jit, static_argnames=())
def alloc_hypothesis_2(
    temp_day, gpp_day, leaf_rdark_day,
    croot, cleaf, cstem, cgrain,
    litter_cleaf_in, grain_fill,
    cratio_resp, cratio_leaf, cratio_root, cratio_biomass,
    turnover_cleaf, turnover_croot, sla, q10, invert_option,
    management_type, management_c_input, management_c_output,
    pft_is_oat, pheno_stage,
):
    """Compute daily biomass allocation from GPP to carbon pools.

    Fortran reference: subroutine alloc_hypothesis_2 in allocation.f90

    Args:
        temp_day: Daily mean temperature (°C).
        gpp_day: Gross primary production (kg C m⁻² s⁻¹).
        leaf_rdark_day: Leaf dark respiration (kg C m⁻² s⁻¹).
        croot: Root carbon (kg C m⁻²).
        cleaf: Leaf carbon (kg C m⁻²).
        cstem: Stem carbon (kg C m⁻²).
        cgrain: Grain carbon (kg C m⁻²).
        litter_cleaf_in: Input leaf litter carbon (used when invert_option != 0).
        grain_fill: Grain-filling flux (kg C m⁻² s⁻¹).
        cratio_resp: Maintenance respiration fraction.
        cratio_leaf: Leaf allocation fraction.
        cratio_root: Root allocation fraction.
        cratio_biomass: Carbon fraction of biomass.
        turnover_cleaf: Leaf turnover rate at 20°C.
        turnover_croot: Root/stem turnover rate at 20°C.
        sla: Specific leaf area (m² kg⁻¹).
        q10: Q10 temperature coefficient.
        invert_option: Inversion mode (0=none, 1=cratio_leaf, 2=turnover).
        management_type: Management type (0=none, 1=harvest, 3=grazing, 4=organic).
        management_c_input: Carbon input rate from management (kg C m⁻² s⁻¹).
        management_c_output: Carbon output rate from management (kg C m⁻² s⁻¹).
        pft_is_oat: 1.0 for oat, 0.0 for grass.
        pheno_stage: Phenological stage (1=growth, 2=dormancy).

    Returns:
        AllocHypothesisResult with fields: npp_day, auto_resp, croot, cleaf,
        cstem, cgrain, litter_cleaf, litter_croot, compost, lai,
        abovebiomass, belowbiomass, yield_, grain_fill, pheno_stage.
    """
    spd = _SPD
    is_growth = pheno_stage == 1
    inv_is_0 = invert_option == 0.0

    # ---------- GROWTH PHASE (pheno_stage == 1) ----------

    # Q10 temperature factor
    q10f = q10 ** ((temp_day - 20.0) / 10.0)

    # Growth respirations
    gr_resp_leaf = jnp.maximum(
        0.0, (gpp_day * cratio_leaf - leaf_rdark_day) * spd) * 0.11
    gr_resp_stem = jnp.maximum(
        (gpp_day * (1.0 - cratio_leaf - cratio_root)
         - cstem * cratio_resp * q10f) * spd, 0.0) * 0.11
    gr_resp_root = jnp.maximum(
        (gpp_day * cratio_root - grain_fill
         - croot * cratio_resp * q10f) * spd, 0.0) * 0.11
    gr_resp_grain = jnp.where(
        pft_is_oat,
        jnp.maximum(
            (grain_fill - cgrain * 0.1 * cratio_resp * q10f) * spd,
            0.0) * 0.11,
        0.0)

    # NPP and autotrophic respiration
    maint_sc = (croot + cstem) * cratio_resp * q10f
    maint_g = cgrain * 0.1 * cratio_resp * q10f
    npp_g = (gpp_day - maint_sc - maint_g - leaf_rdark_day) * spd
    auto_resp_g = (maint_sc + maint_g + leaf_rdark_day) * spd \
        + gr_resp_leaf + gr_resp_stem + gr_resp_root + gr_resp_grain

    # Turnover-based litter
    litter_cleaf_turnover = cleaf * turnover_cleaf * q10f
    litter_cleaf_g = jnp.where(inv_is_0, litter_cleaf_turnover, litter_cleaf_in)
    litter_cstem_g = cstem * turnover_croot * q10f
    litter_croot_g = croot * turnover_croot * q10f
    compost_g = jnp.float64(0.0)

    # Update pools (before management)
    cleaf_g = jnp.where(
        inv_is_0,
        cleaf + gpp_day * cratio_leaf * spd
        - litter_cleaf_g - leaf_rdark_day * spd - gr_resp_leaf,
        cleaf)
    cstem_g = (cstem + gpp_day * (1.0 - cratio_leaf - cratio_root) * spd
               - cstem * cratio_resp * q10f * spd
               - litter_cstem_g - gr_resp_stem)
    croot_g = jnp.maximum(
        0.0,
        croot + (gpp_day * cratio_root - grain_fill) * spd - litter_croot_g
        - croot * cratio_resp * q10f * spd - gr_resp_root)
    cgrain_g = jnp.where(
        pft_is_oat,
        cgrain + grain_fill * spd
        - cgrain * 0.1 * cratio_resp * q10f * spd - gr_resp_grain,
        0.0)

    # --- Management ---
    def _safe_denom(x):
        """Guard the undefined x/0 case with a tiny sign-preserving fallback."""
        return jnp.where(jnp.abs(x) < 1e-30, jnp.copysign(1e-30, x), x)

    # Harvest grass: proportional deduction from cleaf (if inv==0) and cstem
    hg_deduct = management_c_output * spd
    hg_cleaf_frac = hg_deduct * cleaf_g / _safe_denom(cleaf_g + cstem_g)
    hg_cleaf = jnp.where(inv_is_0, cleaf_g - hg_cleaf_frac, cleaf_g)
    # cstem uses possibly-modified cleaf in denominator (sequential effect)
    hg_cstem = cstem_g - hg_deduct * cstem_g / _safe_denom(hg_cleaf + cstem_g)

    # Harvest oat: overwrite litter with updated pools, zero everything
    ho_litter_cleaf = cleaf_g
    ho_litter_croot = croot_g
    ho_litter_cstem = cstem_g

    # Harvest combined (select by pft)
    h_cleaf = jnp.where(pft_is_oat, 0.0, hg_cleaf)
    h_cstem = jnp.where(pft_is_oat, 0.0, hg_cstem)
    h_croot = jnp.where(pft_is_oat, 0.0, croot_g)
    h_cgrain = jnp.where(pft_is_oat, 0.0, cgrain_g)
    h_npp = jnp.where(pft_is_oat, 0.0, npp_g)
    h_resp = jnp.where(pft_is_oat, 0.0, auto_resp_g)
    h_litter_cleaf = jnp.where(pft_is_oat, ho_litter_cleaf, litter_cleaf_g)
    h_litter_croot = jnp.where(pft_is_oat, ho_litter_croot, litter_croot_g)
    h_litter_cstem = jnp.where(pft_is_oat, ho_litter_cstem, litter_cstem_g)
    h_compost = compost_g

    # Grazing: deduct from cleaf (if inv==0) and cstem, add compost
    gz_deduct = management_c_output * spd
    gz_cleaf_frac = gz_deduct * cleaf_g / _safe_denom(cleaf_g + cstem_g)
    gz_cleaf = jnp.where(inv_is_0, cleaf_g - gz_cleaf_frac, cleaf_g)
    gz_cstem = cstem_g - gz_deduct * cstem_g / _safe_denom(gz_cleaf + cstem_g)
    gz_compost = management_c_input * spd

    # Organic: only compost
    org_compost = management_c_input * spd

    # No management: pass through
    is_harvest = management_type == 1
    is_grazing = management_type == 3
    is_organic = management_type == 4

    def _sel(h, gz, org, nm):
        """Select value by management type."""
        return jnp.where(is_harvest, h,
               jnp.where(is_grazing, gz,
               jnp.where(is_organic, org, nm)))

    m_cleaf = _sel(h_cleaf, gz_cleaf, cleaf_g, cleaf_g)
    m_cstem = _sel(h_cstem, gz_cstem, cstem_g, cstem_g)
    m_croot = _sel(h_croot, croot_g, croot_g, croot_g)
    m_cgrain = _sel(h_cgrain, cgrain_g, cgrain_g, cgrain_g)
    m_npp = _sel(h_npp, npp_g, npp_g, npp_g)
    m_resp = _sel(h_resp, auto_resp_g, auto_resp_g, auto_resp_g)
    m_litter_cleaf = _sel(h_litter_cleaf, litter_cleaf_g, litter_cleaf_g, litter_cleaf_g)
    m_litter_croot = _sel(h_litter_croot, litter_croot_g, litter_croot_g, litter_croot_g)
    m_litter_cstem = _sel(h_litter_cstem, litter_cstem_g, litter_cstem_g, litter_cstem_g)
    m_compost = _sel(h_compost, gz_compost, org_compost, compost_g)

    # Combine leaf + stem litter
    g_litter_cleaf = m_litter_cleaf + m_litter_cstem
    g_litter_croot = m_litter_croot

    # Derived quantities (growth phase)
    g_above = (m_cleaf + m_cstem + m_cgrain) / cratio_biomass
    g_below = m_croot / cratio_biomass
    g_lai = m_cleaf / cratio_biomass * sla
    g_pheno = 1  # unchanged in growth

    # ---------- DORMANCY PHASE (pheno_stage == 2) ----------
    d_litter_cleaf = cleaf   # original input value (NOT combined with cstem)
    d_litter_croot = croot   # original input value
    d_above = cgrain / cratio_biomass  # only cgrain survives (not zeroed)
    d_below = jnp.float64(0.0)
    d_lai = jnp.float64(0.0)
    d_pheno = 1  # reset to growth

    # ---------- SELECT BY PHENOLOGICAL STAGE ----------
    return AllocHypothesisResult(
        npp_day=jnp.where(is_growth, m_npp, 0.0),
        auto_resp=jnp.where(is_growth, m_resp, 0.0),
        croot=jnp.where(is_growth, m_croot, 0.0),
        cleaf=jnp.where(is_growth, m_cleaf, 0.0),
        cstem=jnp.where(is_growth, m_cstem, 0.0),
        cgrain=jnp.where(is_growth, m_cgrain, cgrain),
        litter_cleaf=jnp.where(is_growth, g_litter_cleaf, d_litter_cleaf),
        litter_croot=jnp.where(is_growth, g_litter_croot, d_litter_croot),
        compost=jnp.where(is_growth, m_compost, 0.0),
        lai=jnp.where(is_growth, g_lai, d_lai),
        abovebiomass=jnp.where(is_growth, g_above, d_above),
        belowbiomass=jnp.where(is_growth, g_below, d_below),
        yield_=jnp.float64(0.0),  # never modified
        grain_fill=grain_fill,    # never modified
        pheno_stage=jnp.where(is_growth, g_pheno, d_pheno),
    )
