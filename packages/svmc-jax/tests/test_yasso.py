"""Tests for Yasso20 leaf functions in JAX — validated against Fortran reference."""

import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from svmc_jax.yasso.leaf_functions import inputs_to_fractions
from svmc_jax.yasso.matrixexp import matrixexp, matrixnorm

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
