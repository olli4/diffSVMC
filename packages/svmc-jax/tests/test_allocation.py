"""Tests for allocation module (alloc_hypothesis_2 and invert_alloc)."""

import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from svmc_jax.allocation import alloc_hypothesis_2, invert_alloc

FIXTURES_DIR = Path(__file__).resolve().parent / "../../svmc-ref/fixtures"
ALLOC = json.loads((FIXTURES_DIR / "allocation.json").read_text())

RTOL = 1e-10


def _pft_flag(pft_type: str) -> float:
    """Convert pft_type string to numeric flag (0.0=grass, 1.0=oat)."""
    return 1.0 if pft_type == "oat" else 0.0


# ──────────────────────────────────────────────────────────────────
# alloc_hypothesis_2 — fixture playback
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "c", ALLOC["alloc_hypothesis_2"],
    ids=lambda c: (
        f"pheno={c['inputs']['pheno_stage']}"
        f"_inv={int(c['inputs']['invert_option'])}"
        f"_mgmt={c['inputs']['management_type']}"
        f"_pft={c['inputs']['pft_type']}"
    ),
)
def test_alloc_hypothesis_2(c):
    inp = c["inputs"]
    out = c["output"]

    result = alloc_hypothesis_2(
        temp_day=inp["temp_day"],
        gpp_day=inp["gpp_day"],
        leaf_rdark_day=inp["leaf_rdark_day"],
        croot=inp["croot"],
        cleaf=inp["cleaf"],
        cstem=inp["cstem"],
        cgrain=inp["cgrain"],
        litter_cleaf_in=inp["litter_cleaf"],
        grain_fill=inp["grain_fill"],
        cratio_resp=inp["cratio_resp"],
        cratio_leaf=inp["cratio_leaf"],
        cratio_root=inp["cratio_root"],
        cratio_biomass=inp["cratio_biomass"],
        turnover_cleaf=inp["turnover_cleaf"],
        turnover_croot=inp["turnover_croot"],
        sla=inp["sla"],
        q10=inp["q10"],
        invert_option=inp["invert_option"],
        management_type=inp["management_type"],
        management_c_input=inp["management_c_input"],
        management_c_output=inp["management_c_output"],
        pft_is_oat=_pft_flag(inp["pft_type"]),
        pheno_stage=inp["pheno_stage"],
    )

    for key in out:
        expected = out[key]
        actual = float(result[key])
        if abs(expected) < 1e-15:
            assert abs(actual) < 1e-12, f"{key}: expected ~0, got {actual}"
        else:
            assert jnp.allclose(
                jnp.array(actual), jnp.array(expected), rtol=RTOL
            ), f"{key}: expected {expected}, got {actual}"


# ──────────────────────────────────────────────────────────────────
# invert_alloc — fixture playback
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "c", ALLOC["invert_alloc"],
    ids=lambda c: (
        f"pheno={c['inputs']['pheno_stage']}"
        f"_inv={int(c['inputs']['invert_option'])}"
        f"_mgmt={c['inputs']['management_type']}"
        f"_pft={c['inputs']['pft_type']}"
    ),
)
def test_invert_alloc(c):
    inp = c["inputs"]
    out = c["output"]

    result = invert_alloc(
        delta_lai=inp["delta_lai"],
        leaf_rdark_day=inp["leaf_rdark_day"],
        temp_day=inp["temp_day"],
        gpp_day=inp["gpp_day"],
        litter_cleaf_in=inp["litter_cleaf"],
        cleaf=inp["cleaf"],
        cstem=inp["cstem"],
        cratio_resp=inp["cratio_resp"],
        cratio_leaf=inp["cratio_leaf"],
        cratio_root=inp["cratio_root"],
        cratio_biomass=inp["cratio_biomass"],
        harvest_index=inp["harvest_index"],
        turnover_cleaf=inp["turnover_cleaf"],
        turnover_croot=inp["turnover_croot"],
        sla=inp["sla"],
        q10=inp["q10"],
        invert_option=inp["invert_option"],
        management_type=inp["management_type"],
        management_c_input=inp["management_c_input"],
        management_c_output=inp["management_c_output"],
        pft_is_oat=_pft_flag(inp["pft_type"]),
        pheno_stage=inp["pheno_stage"],
    )

    for key in out:
        expected = out[key]
        actual = float(result[key])
        if abs(expected) < 1e-15:
            assert abs(actual) < 1e-12, f"{key}: expected ~0, got {actual}"
        else:
            assert jnp.allclose(
                jnp.array(actual), jnp.array(expected), rtol=RTOL
            ), f"{key}: expected {expected}, got {actual}"


# ──────────────────────────────────────────────────────────────────
# Invariant tests
# ──────────────────────────────────────────────────────────────────

def test_ah2_dormancy_zeroes_pools():
    """Dormancy (pheno_stage=2) must zero all living carbon pools."""
    result = alloc_hypothesis_2(
        temp_day=15.0, gpp_day=3e-7, leaf_rdark_day=3e-8,
        croot=0.05, cleaf=0.10, cstem=0.02, cgrain=0.0,
        litter_cleaf_in=0.0, grain_fill=0.0,
        cratio_resp=0.4, cratio_leaf=0.8, cratio_root=0.2,
        cratio_biomass=0.42, turnover_cleaf=0.41/365,
        turnover_croot=0.41/365, sla=10.0, q10=2.0,
        invert_option=0.0, management_type=0,
        management_c_input=0.0, management_c_output=0.0,
        pft_is_oat=0.0, pheno_stage=2,
    )
    assert float(result["croot"]) == 0.0
    assert float(result["cleaf"]) == 0.0
    assert float(result["cstem"]) == 0.0
    assert float(result["npp_day"]) == 0.0
    assert float(result["auto_resp"]) == 0.0


def test_ah2_resp_budget():
    """auto_resp should exceed maintenance-only (growth resp is non-negative)."""
    result = alloc_hypothesis_2(
        temp_day=20.0, gpp_day=3e-7, leaf_rdark_day=3e-8,
        croot=0.05, cleaf=0.10, cstem=0.02, cgrain=0.0,
        litter_cleaf_in=0.0, grain_fill=0.0,
        cratio_resp=0.02, cratio_leaf=0.8, cratio_root=0.2,
        cratio_biomass=0.42, turnover_cleaf=0.41/365,
        turnover_croot=0.41/365, sla=10.0, q10=1.0,
        invert_option=0.0, management_type=0,
        management_c_input=0.0, management_c_output=0.0,
        pft_is_oat=0.0, pheno_stage=1,
    )
    assert float(result["auto_resp"]) > 0.0, "auto_resp should be positive"
    # NPP + auto_resp >= GPP*SPD (growth respiration is non-negative)
    gpp_daily = 3e-7 * 86400.0
    total = float(result["npp_day"]) + float(result["auto_resp"])
    assert total >= gpp_daily - 1e-15, (
        f"npp+resp={total} should >= gpp_daily={gpp_daily}"
    )


def test_ah2_harvest_grass_uses_updated_cleaf_in_second_denominator():
    """Harvest grass must reuse the post-cleaf-update denominator for cstem."""
    out = alloc_hypothesis_2(
        temp_day=20.0, gpp_day=1e-7, leaf_rdark_day=1e-8,
        croot=0.10, cleaf=0.02, cstem=0.01, cgrain=0.0,
        litter_cleaf_in=0.0, grain_fill=0.0,
        cratio_resp=0.0, cratio_leaf=0.4, cratio_root=0.2,
        cratio_biomass=0.42, turnover_cleaf=0.001,
        turnover_croot=0.001, sla=10.0, q10=1.0,
        invert_option=0.0, management_type=1,
        management_c_input=0.0, management_c_output=1e-5,
        pft_is_oat=0.0, pheno_stage=1,
    )

    spd = 86400.0
    q10f = 1.0
    gr_resp_leaf = max(0.0, (1e-7 * 0.4 - 1e-8) * spd) * 0.11
    gr_resp_stem = max((1e-7 * (1.0 - 0.4 - 0.2) - 0.01 * 0.0 * q10f) * spd, 0.0) * 0.11
    litter_cleaf = 0.02 * 0.001 * q10f
    litter_cstem = 0.01 * 0.001 * q10f
    cleaf_g = 0.02 + 1e-7 * 0.4 * spd - litter_cleaf - 1e-8 * spd - gr_resp_leaf
    cstem_g = 0.01 + 1e-7 * (1.0 - 0.4 - 0.2) * spd - 0.01 * 0.0 * q10f * spd - litter_cstem - gr_resp_stem
    deduct = 1e-5 * spd

    cleaf_after = cleaf_g - deduct * cleaf_g / (cleaf_g + cstem_g)
    expected_cstem = cstem_g - deduct * cstem_g / (cleaf_after + cstem_g)
    old_denom_cstem = cstem_g - deduct * cstem_g / (cleaf_g + cstem_g)

    assert jnp.allclose(out["cstem"], expected_cstem, rtol=RTOL)
    assert not jnp.allclose(out["cstem"], old_denom_cstem, rtol=RTOL)


def test_ah2_grazing_inv_false_keeps_cleaf_but_updates_cstem_and_compost():
    """Grazing with invert_option!=0 must skip cleaf deduction but still graze cstem."""
    out = alloc_hypothesis_2(
        temp_day=15.0, gpp_day=3e-7, leaf_rdark_day=3e-8,
        croot=0.05, cleaf=0.10, cstem=0.02, cgrain=0.0,
        litter_cleaf_in=0.0, grain_fill=0.0,
        cratio_resp=0.4, cratio_leaf=0.8, cratio_root=0.2,
        cratio_biomass=0.42, turnover_cleaf=0.41/365,
        turnover_croot=0.41/365, sla=10.0, q10=2.0,
        invert_option=1.0, management_type=3,
        management_c_input=1e-8, management_c_output=5e-8,
        pft_is_oat=0.0, pheno_stage=1,
    )

    assert jnp.allclose(out["cleaf"], 0.1, rtol=RTOL)
    assert float(out["cstem"]) != 0.02
    assert jnp.allclose(out["compost"], 1e-8 * 86400.0, rtol=RTOL)


def test_ah2_differentiable():
    """alloc_hypothesis_2 must be differentiable w.r.t. gpp_day."""
    def f(gpp):
        out = alloc_hypothesis_2(
            temp_day=15.0, gpp_day=gpp, leaf_rdark_day=3e-8,
            croot=0.05, cleaf=0.10, cstem=0.02, cgrain=0.0,
            litter_cleaf_in=0.0, grain_fill=0.0,
            cratio_resp=0.4, cratio_leaf=0.8, cratio_root=0.2,
            cratio_biomass=0.42, turnover_cleaf=0.41/365,
            turnover_croot=0.41/365, sla=10.0, q10=2.0,
            invert_option=0.0, management_type=0,
            management_c_input=0.0, management_c_output=0.0,
            pft_is_oat=0.0, pheno_stage=1,
        )
        return out["npp_day"]

    grad_fn = jax.grad(f)
    g = grad_fn(3e-7)
    assert jnp.isfinite(g), f"Gradient is not finite: {g}"
    assert float(g) != 0.0, "Gradient is zero"


def test_invert_alloc_differentiable():
    """invert_alloc must be differentiable w.r.t. delta_lai."""
    def f(dlai):
        out = invert_alloc(
            delta_lai=dlai, leaf_rdark_day=3e-8, temp_day=15.0,
            gpp_day=3e-7, litter_cleaf_in=0.0, cleaf=0.10, cstem=0.02,
            cratio_resp=0.4, cratio_leaf=0.8, cratio_root=0.2,
            cratio_biomass=0.42, harvest_index=0.5,
            turnover_cleaf=0.41/365, turnover_croot=0.41/365,
            sla=10.0, q10=2.0, invert_option=1.0,
            management_type=0, management_c_input=0.0,
            management_c_output=0.0, pft_is_oat=0.0, pheno_stage=1,
        )
        return out["cratio_leaf"]

    grad_fn = jax.grad(f)
    g = grad_fn(0.05)
    assert jnp.isfinite(g), f"Gradient is not finite: {g}"
    assert float(g) != 0.0, "Gradient is zero"
