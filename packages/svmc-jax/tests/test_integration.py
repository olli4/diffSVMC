"""Phase 5 integration replay test.

Plays back the 35-day Qvidja cold-start reference replay through the
composed JAX model (P-Hydro → canopy_water_flux → soil_water →
allocation → Yasso) and compares daily outputs against the Fortran
reference fixture.
"""

import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from svmc_jax.integration import run_integration, run_integration_grouped
from svmc_jax.qvidja_replay import build_qvidja_run_inputs, build_qvidja_run_kwargs

# ── Paths ─────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_FIXTURES_DIR = _HERE / "../../svmc-ref/fixtures"
_QVIDJA_REF = _HERE / "../../../website/public/qvidja-v1-reference.json"

# Integration fixture tolerance.
#
# The JAX L-BFGS-B optimizer (jaxopt.LBFGSB) and the Fortran L-BFGS-B
# optimizer find the same economic optimum to within the documented
# per-call tolerance of rtol=3e-3 (see test_phydro.py::rtol_solver).
# Over a 35-day integration with ~20 light-hour optimizer calls per day,
# the per-day GPP error stays below 1e-2 (max observed: 9e-3).
# Water-balance fields (wliq, psi) are insensitive to the optimizer
# and match to ~1e-3.
#
# For NEE (= respiration − GPP), the two large terms nearly cancel,
# making the relative error of their difference much larger than either
# component.  We therefore use an absolute tolerance for NEE derived
# from the GPP-scale error.
RTOL = 1e-2
ATOL = 1e-12

# NEE tolerance: absolute, scaled to the GPP error magnitude.
# max(gpp_avg) ≈ 1e-7, RTOL * that ≈ 1e-9
NEE_ATOL = 1e-9

# ── Data loading ──────────────────────────────────────────────────────

_NDAYS = 35
_NHOURS = _NDAYS * 24


def _load_qvidja_ref():
    with open(_QVIDJA_REF) as f:
        return json.load(f)


def _load_integration_fixture():
    with open(_FIXTURES_DIR / "integration.json") as f:
        data = json.load(f)
    return data["integration_daily"]


# ── Test ──────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_integration_35day_replay():
    """Play back the 35-day Qvidja replay through the composed JAX model."""

    ref = _load_qvidja_ref()
    fixture = _load_integration_fixture()
    assert len(fixture) == _NDAYS

    # Run the composed integration
    _final_carry, daily_outputs = run_integration(**build_qvidja_run_kwargs(ref, _NDAYS))

    # Compare per-day outputs against the Fortran reference fixture
    scalar_keys = [
        "gpp_avg", "nee", "hetero_resp", "auto_resp",
        "cleaf", "croot", "cstem", "cgrain",
        "lai_alloc", "litter_cleaf", "litter_croot",
        "soc_total", "wliq", "psi",
    ]

    for day_idx in range(_NDAYS):
        expected = fixture[day_idx]["output"]
        day_label = f"day {day_idx + 1}"

        for key in scalar_keys:
            exp_val = expected[key]
            # Access the corresponding DailyOutput field
            actual_arr = getattr(daily_outputs, key)
            actual_val = float(actual_arr[day_idx])

            # NEE and auto/hetero respiration are near-cancellation
            # quantities; use absolute tolerance instead of relative.
            if key in ("nee", "hetero_resp", "auto_resp"):
                assert abs(actual_val - exp_val) < NEE_ATOL, (
                    f"{day_label} {key}: expected {exp_val}, got {actual_val} "
                    f"(abs_err={abs(actual_val - exp_val):.2e})"
                )
            elif abs(exp_val) < ATOL:
                assert abs(actual_val - exp_val) < ATOL, (
                    f"{day_label} {key}: expected {exp_val}, got {actual_val}"
                )
            else:
                rel_err = abs(actual_val - exp_val) / abs(exp_val)
                assert rel_err < RTOL, (
                    f"{day_label} {key}: expected {exp_val}, got {actual_val} "
                    f"(rel_err={rel_err:.2e})"
                )

        # AWENH state vector
        exp_cstate = np.array(expected["cstate"])
        actual_cstate = np.array(daily_outputs.cstate[day_idx])
        np.testing.assert_allclose(
            actual_cstate, exp_cstate, rtol=RTOL, atol=ATOL,
            err_msg=f"{day_label} cstate mismatch",
        )

    # ── Summary-level derived metrics (PLAN.md Lesson 3) ──
    # Cumulative sums catch drift that per-day checks miss.
    ref_gpp_sum = sum(f["output"]["gpp_avg"] for f in fixture)
    ref_nee_sum = sum(f["output"]["nee"] for f in fixture)
    ref_final_soc = fixture[-1]["output"]["soc_total"]
    ref_final_wliq = fixture[-1]["output"]["wliq"]

    actual_gpp_sum = float(jnp.sum(daily_outputs.gpp_avg))
    actual_nee_sum = float(jnp.sum(daily_outputs.nee))
    actual_final_soc = float(daily_outputs.soc_total[-1])
    actual_final_wliq = float(daily_outputs.wliq[-1])

    np.testing.assert_allclose(
        actual_gpp_sum, ref_gpp_sum, rtol=RTOL,
        err_msg="35-day cumulative GPP mismatch",
    )
    assert abs(actual_nee_sum - ref_nee_sum) < NEE_ATOL * _NDAYS, (
        f"35-day cumulative NEE: expected {ref_nee_sum:.4e}, "
        f"got {actual_nee_sum:.4e}"
    )
    np.testing.assert_allclose(
        actual_final_soc, ref_final_soc, rtol=RTOL,
        err_msg="Final SOC mismatch",
    )
    np.testing.assert_allclose(
        actual_final_wliq, ref_final_wliq, rtol=RTOL,
        err_msg="Final soil moisture mismatch",
    )

    # ── Harvest event validation (day 34 = manage_type 1) ──
    # The fixture includes a harvest at day 34 (0-indexed 33).
    # Verify the integration reproduced its impact on carbon pools.
    harvest_day = 33  # 0-indexed
    ref_harvest_cleaf = fixture[harvest_day]["output"]["cleaf"]
    actual_harvest_cleaf = float(daily_outputs.cleaf[harvest_day])
    np.testing.assert_allclose(
        actual_harvest_cleaf, ref_harvest_cleaf, rtol=RTOL,
        err_msg="Harvest day cleaf mismatch",
    )


def _run_1day(defaults, hourly, daily, **overrides):
    """Run a 1-day integration with Qvidja defaults, returning outputs.

    Keyword overrides are applied to the hourly forcing arrays *after*
    the default slice-and-reshape (e.g. ``hourly_temp=custom_array``).
    """
    kw = build_qvidja_run_kwargs(
        {"defaults": defaults, "hourly": hourly, "daily": daily},
        1,
    )
    kw.update(overrides)
    return run_integration(**kw)


def test_integration_1day_differentiable_through_phydro():
    """A one-day integration slice should remain differentiable."""
    ref = _load_qvidja_ref()
    defaults = ref["defaults"]

    def loss(alpha_cost):
        _fc, out = _run_1day(defaults, ref["hourly"], ref["daily"],
                             alpha_cost=alpha_cost)
        return out.gpp_avg[0]

    g = jax.grad(loss)(jnp.array(defaults["alpha"]))
    assert jnp.isfinite(g), f"Gradient through one-day integration is not finite: {g}"


@pytest.mark.parametrize("tc_offset,vpd_scale,label", [
    (45.0, 1.0, "hot"),      # tc ≈ 45 °C
    (-40.0, 1.0, "cold"),    # tc ≈ -40 °C
    (0.0, 5.0, "high_vpd"),  # VPD × 5 (very dry)
])
def test_integration_1day_ood_gradients(tc_offset, vpd_scale, label):
    """OOD gradients through the integration loop should remain finite.

    DoD criterion 5: verify ``jax.grad`` produces finite, stable
    gradients under adversarial forcing conditions that L-BFGS-B
    inversion could encounter.
    """
    ref = _load_qvidja_ref()
    defaults = ref["defaults"]
    hourly = ref["hourly"]
    daily = ref["daily"]

    # Build OOD forcing: shift temperature or scale VPD.
    hourly_temp = jnp.array(hourly["temp_hr"][:24]).reshape(1, 24) + tc_offset
    hourly_vpd = jnp.array(hourly["vpd_hr"][:24]).reshape(1, 24) * vpd_scale

    def loss(alpha_cost):
        _fc, out = _run_1day(defaults, hourly, daily,
                             hourly_temp=hourly_temp,
                             hourly_vpd=hourly_vpd,
                             alpha_cost=alpha_cost)
        return out.gpp_avg[0]

    g = jax.grad(loss)(jnp.array(defaults["alpha"]))
    assert jnp.isfinite(g), (
        f"OOD gradient ({label}) is not finite: {g}"
    )


def test_integration_1day_jit_consistent():
    """JIT-compiled integration must produce the same outputs as eager.

    Guards against a future change breaking the compiled execution path
    that calibration/inversion workloads will rely on.
    """
    ref = _load_qvidja_ref()
    defaults = ref["defaults"]

    _fc_eager, out_eager = _run_1day(defaults, ref["hourly"], ref["daily"])

    @jax.jit
    def run_jitted(alpha):
        return _run_1day(defaults, ref["hourly"], ref["daily"],
                         alpha_cost=alpha)

    _fc_jit, out_jit = run_jitted(jnp.array(defaults["alpha"]))

    np.testing.assert_allclose(
        float(out_jit.gpp_avg[0]), float(out_eager.gpp_avg[0]),
        rtol=1e-9,
        atol=5e-16,
        err_msg="JIT vs eager GPP mismatch",
    )
    np.testing.assert_allclose(
        float(out_jit.et_total[0]), float(out_eager.et_total[0]),
        rtol=1e-9,
        atol=1e-8,
        err_msg="JIT vs eager ET mismatch",
    )


def test_integration_1day_projected_newton_selectable():
    """The integration entry point should allow the projected-Newton alternative."""
    ref = _load_qvidja_ref()
    defaults = ref["defaults"]

    _fc, out = _run_1day(
        defaults, ref["hourly"], ref["daily"],
        phydro_optimizer="projected_newton",
    )

    assert jnp.isfinite(out.gpp_avg[0]), f"Projected-Newton GPP is not finite: {out.gpp_avg[0]}"
    assert jnp.isfinite(out.et_total[0]), f"Projected-Newton ET is not finite: {out.et_total[0]}"


def test_integration_1day_et_nonnegative():
    """Daily ET must be non-negative (physical invariant)."""
    ref = _load_qvidja_ref()
    defaults = ref["defaults"]

    _fc, out = _run_1day(defaults, ref["hourly"], ref["daily"])
    et = float(out.et_total[0])
    assert jnp.isfinite(out.et_total[0]), f"ET is not finite: {et}"
    assert et >= 0.0, f"ET is negative: {et}"


def test_integration_grouped_entrypoint_vmappable():
    """Grouped integration inputs should support site batching via vmap."""
    ref = _load_qvidja_ref()
    forcing, params = build_qvidja_run_inputs(ref, 1)

    _single_carry, single_out = run_integration_grouped(forcing, params)

    batched_forcing = jax.tree.map(lambda x: jnp.stack([x, x]), forcing)
    batched_params = jax.tree.map(lambda x: jnp.stack([x, x]), params)

    batched_run = jax.jit(
        jax.vmap(
            lambda forcing_i, params_i: run_integration_grouped(forcing_i, params_i),
        )
    )
    _batched_carry, batched_out = batched_run(batched_forcing, batched_params)

    expected_gpp = float(single_out.gpp_avg[0])
    expected_et = float(single_out.et_total[0])
    expected_soc = float(single_out.soc_total[0])

    np.testing.assert_allclose(
        np.array(batched_out.gpp_avg[:, 0]),
        np.full((2,), expected_gpp),
        rtol=1e-9,
        atol=5e-16,
        err_msg="vmapped grouped GPP mismatch",
    )
    np.testing.assert_allclose(
        np.array(batched_out.et_total[:, 0]),
        np.full((2,), expected_et),
        rtol=1e-9,
        atol=1e-8,
        err_msg="vmapped grouped ET mismatch",
    )
    np.testing.assert_allclose(
        np.array(batched_out.soc_total[:, 0]),
        np.full((2,), expected_soc),
        rtol=1e-9,
        err_msg="vmapped grouped SOC mismatch",
    )
