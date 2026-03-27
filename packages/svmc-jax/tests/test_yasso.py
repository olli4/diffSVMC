"""Tests for Yasso20 leaf functions in JAX — validated against Fortran reference."""

import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from svmc_jax.yasso.leaf_functions import inputs_to_fractions
from svmc_jax.yasso.matrixexp import matrixexp, matrixnorm
from svmc_jax.yasso.mod5c20 import mod5c20

FIXTURES_DIR = Path(__file__).resolve().parent / "../../svmc-ref/fixtures"
YASSO = json.loads((FIXTURES_DIR / "yasso.json").read_text())

RTOL = 1e-10


# ── inputs_to_fractions ──────────────────────────────────────────────


@pytest.mark.parametrize("c", YASSO["inputs_to_fractions"],
    ids=lambda c: f"leaf={c['inputs']['leaf']}_root={c['inputs']['root']}"
                  f"_sol={c['inputs']['soluble']}_comp={c['inputs']['compost']}")
def test_inputs_to_fractions(c):
    inp = c["inputs"]
    result = inputs_to_fractions(inp["leaf"], inp["root"], inp["soluble"], inp["compost"])
    assert jnp.allclose(result, jnp.array(c["output"]), rtol=RTOL)


# ── Invariants ───────────────────────────────────────────────────────


def test_inputs_to_fractions_h_pool_zero():
    """The H (humus) pool should never receive external input."""
    result = inputs_to_fractions(1.0, 1.0, 1.0, 1.0)
    assert float(result[4]) == 0.0


def test_inputs_to_fractions_sums_to_total():
    """Total C in AWENH fractions should equal total C input (mass conservation)."""
    leaf, root, sol, comp = 0.5, 0.3, 0.1, 0.1
    result = inputs_to_fractions(leaf, root, sol, comp)
    total_in = leaf + root + sol + comp
    assert jnp.allclose(jnp.sum(result), total_in, rtol=1e-10)


def test_inputs_to_fractions_linearity():
    """AWENH decomposition is linear: f(2x) = 2·f(x)."""
    r1 = inputs_to_fractions(1.0, 0.5, 0.2, 0.3)
    r2 = inputs_to_fractions(2.0, 1.0, 0.4, 0.6)
    assert jnp.allclose(r2, 2.0 * r1, rtol=1e-12)


# ── matrixnorm ───────────────────────────────────────────────────────


@pytest.mark.parametrize("c", YASSO["matrixnorm"],
    ids=lambda c: "diag=" + "_".join(f"{c['inputs']['a'][i][i]:.2f}" for i in range(5)))
def test_matrixnorm(c):
    a = jnp.array(c["inputs"]["a"])
    result = matrixnorm(a)
    assert jnp.allclose(result, c["output"], rtol=RTOL)


# ── matrixexp ────────────────────────────────────────────────────────


@pytest.mark.parametrize("c", YASSO["matrixexp"],
    ids=lambda c: "diag=" + "_".join(f"{c['inputs']['a'][i][i]:.2f}" for i in range(5)))
def test_matrixexp(c):
    a = jnp.array(c["inputs"]["a"])
    expected = jnp.array(c["output"])
    result = matrixexp(a)
    assert jnp.allclose(result, expected, rtol=RTOL)


# ── matrixexp invariants ─────────────────────────────────────────────


def test_matrixexp_zero_is_identity():
    """exp(0) = I."""
    result = matrixexp(jnp.zeros((5, 5)))
    assert jnp.allclose(result, jnp.eye(5), atol=1e-12)


def test_matrixexp_diagonal():
    """exp(diag(a)) = diag(exp(a))."""
    diag_vals = jnp.array([-0.5, -0.3, -0.2, -0.1, -0.05])
    a = jnp.diag(diag_vals)
    result = matrixexp(a)
    expected = jnp.diag(jnp.exp(diag_vals))
    assert jnp.allclose(result, expected, rtol=1e-6)


def test_matrixexp_jit():
    """matrixexp should be JIT-compilable."""
    jit_matrixexp = jax.jit(matrixexp)
    a = jnp.diag(jnp.array([-0.1, -0.2, -0.3, -0.4, -0.5]))
    result = jit_matrixexp(a)
    expected = matrixexp(a)
    assert jnp.allclose(result, expected, rtol=1e-12)


def test_matrixexp_gradient():
    """Gradients of matrixexp should be finite."""
    a = jnp.diag(jnp.array([-0.1, -0.2, -0.3, -0.4, -0.5]))
    grad_fn = jax.grad(lambda x: jnp.sum(matrixexp(x)))
    g = grad_fn(a)
    assert jnp.all(jnp.isfinite(g))


# ── mod5c20 ──────────────────────────────────────────────────────────


def _mod5c20_id(c):
    ss = c["inputs"].get("steadystate_pred", False)
    tag = "ss" if ss else f"t={c['inputs']['time']:.0f}"
    init_sum = sum(c["inputs"]["init"])
    b_sum = sum(c["inputs"]["b"])
    return f"{tag}_init={init_sum:.1f}_b={b_sum:.2f}"


@pytest.mark.parametrize("c", YASSO["mod5c20"], ids=_mod5c20_id)
def test_mod5c20(c):
    inp = c["inputs"]
    theta = jnp.array(inp["theta"])
    time = jnp.array(inp["time"])
    temp = jnp.array(inp["temp"])
    prec = jnp.array(inp["prec"])
    init = jnp.array(inp["init"])
    b = jnp.array(inp["b"])
    d = jnp.array(inp["d"])
    leac = jnp.array(inp["leac"])
    ss = inp.get("steadystate_pred", False)

    result = mod5c20(theta, time, temp, prec, init, b, d, leac,
                     steadystate_pred=ss)
    expected = jnp.array(c["output"])
    assert jnp.allclose(result, expected, rtol=RTOL)


# ── mod5c20 invariants ───────────────────────────────────────────────


def test_mod5c20_zero_input_decay():
    """With zero input, pools should decrease (pure decay)."""
    # Use case 5 from fixtures: b=0, nonzero init
    c = [x for x in YASSO["mod5c20"]
         if sum(x["inputs"]["b"]) == 0.0][0]
    inp = c["inputs"]
    result = mod5c20(
        jnp.array(inp["theta"]), jnp.array(inp["time"]),
        jnp.array(inp["temp"]), jnp.array(inp["prec"]),
        jnp.array(inp["init"]), jnp.array(inp["b"]),
        jnp.array(inp["d"]), jnp.array(inp["leac"]))
    init = jnp.array(inp["init"])
    # Total carbon should decrease with no input
    assert jnp.sum(result) < jnp.sum(init)


def test_mod5c20_extreme_cold():
    """Extreme cold (-80 °C): tem ≈ 3e-8 > TOL so transient path runs,
    but decomposition is negligible — result ≈ init + b·time."""
    c = [x for x in YASSO["mod5c20"]
         if x["inputs"]["temp"][0] == -80.0][0]
    inp = c["inputs"]
    result = mod5c20(
        jnp.array(inp["theta"]), jnp.array(inp["time"]),
        jnp.array(inp["temp"]), jnp.array(inp["prec"]),
        jnp.array(inp["init"]), jnp.array(inp["b"]),
        jnp.array(inp["d"]), jnp.array(inp["leac"]))
    expected = jnp.array(c["output"])
    assert jnp.allclose(result, expected, rtol=RTOL)


def test_mod5c20_jit():
    """mod5c20 should be JIT-compilable."""
    c = YASSO["mod5c20"][0]
    inp = c["inputs"]
    args = (jnp.array(inp["theta"]), jnp.array(inp["time"]),
            jnp.array(inp["temp"]), jnp.array(inp["prec"]),
            jnp.array(inp["init"]), jnp.array(inp["b"]),
            jnp.array(inp["d"]), jnp.array(inp["leac"]))
    jit_fn = jax.jit(mod5c20)
    result = jit_fn(*args)
    expected = mod5c20(*args)
    assert jnp.allclose(result, expected, rtol=1e-12)


def test_mod5c20_gradient():
    """Gradients of mod5c20 w.r.t. init should be finite."""
    c = YASSO["mod5c20"][0]
    inp = c["inputs"]
    theta = jnp.array(inp["theta"])
    time = jnp.array(inp["time"])
    temp = jnp.array(inp["temp"])
    prec = jnp.array(inp["prec"])
    b = jnp.array(inp["b"])
    d = jnp.array(inp["d"])
    leac = jnp.array(inp["leac"])

    grad_fn = jax.grad(lambda x: jnp.sum(
        mod5c20(theta, time, temp, prec, x, b, d, leac)))
    g = grad_fn(jnp.array(inp["init"]))
    assert jnp.all(jnp.isfinite(g))


# ── mod5c20 Taylor/direct switch boundary ────────────────────────────


def test_mod5c20_norm_switch_boundary():
    """Both sides of the Taylor/direct switch produce consistent results.

    Uses case 0 parameters and scales time so ||At|| lands just below
    and just above sqrt(eps).  Verifies:
      1. Both results are close to init + b·time (leading-order ODE).
      2. The two results are close to each other (smooth transition).
    """
    from svmc_jax.yasso.mod5c20 import _build_coefficient_matrix, _NORM_SWITCH
    from svmc_jax.yasso.matrixexp import matrixnorm

    c = YASSO["mod5c20"][0]
    inp = c["inputs"]
    theta = jnp.array(inp["theta"])
    temp = jnp.array(inp["temp"])
    prec = jnp.array(inp["prec"])
    init = jnp.array(inp["init"])
    b = jnp.array(inp["b"])
    d = jnp.array(inp["d"])
    leac = jnp.array(inp["leac"])

    A = _build_coefficient_matrix(theta, temp, prec, d, leac)
    norm_A = float(matrixnorm(A))

    # Time values that straddle the switch: ||At|| = norm_A * time
    time_below = jnp.array(float(_NORM_SWITCH) / norm_A * 0.5)  # Taylor branch
    time_above = jnp.array(float(_NORM_SWITCH) / norm_A * 2.0)  # direct branch

    r_below = mod5c20(theta, time_below, temp, prec, init, b, d, leac)
    r_above = mod5c20(theta, time_above, temp, prec, init, b, d, leac)

    # Leading-order check: result ≈ init + b·time when time ≈ 0
    approx_below = init + b * time_below
    approx_above = init + b * time_above
    assert jnp.allclose(r_below, approx_below, rtol=1e-6), \
        f"Taylor branch deviates from leading order: {r_below} vs {approx_below}"
    assert jnp.allclose(r_above, approx_above, rtol=1e-6), \
        f"Direct branch deviates from leading order: {r_above} vs {approx_above}"

    # Smoothness: times just across the switch should give nearly identical results
    time_just_below = jnp.array(float(_NORM_SWITCH) / norm_A * 0.99)
    time_just_above = jnp.array(float(_NORM_SWITCH) / norm_A * 1.01)
    r_just_below = mod5c20(theta, time_just_below, temp, prec, init, b, d, leac)
    r_just_above = mod5c20(theta, time_just_above, temp, prec, init, b, d, leac)
    assert jnp.allclose(r_just_below, r_just_above, rtol=0.05), \
        f"Discontinuity at switch: {r_just_below} vs {r_just_above}"
