"""Tests for P-Hydro leaf functions in JAX — validated against Fortran reference."""

import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from svmc_jax.phydro.leaf_functions import (
    ftemp_arrh,
    gammastar,
    ftemp_kphio,
    density_h2o,
    viscosity_h2o,
    calc_kmm,
    scale_conductivity,
    calc_gs,
    calc_assim_light_limited,
    fn_profit,
    quadratic,
    ParEnv,
    ParPlant,
    ParCost,
    ParPhotosynth,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "../../svmc-ref/fixtures"
PHYDRO = json.loads((FIXTURES_DIR / "phydro.json").read_text())

RTOL = 1e-10
KPHIO = 0.087182  # readvegpara_mod default


# ── Simple scalar functions ────────────────────────────────────────────


@pytest.mark.parametrize("c", PHYDRO["ftemp_arrh"],
    ids=lambda c: f"tk={c['inputs']['tk']:.1f}_dha={c['inputs']['dha']:.0f}")
def test_ftemp_arrh(c):
    result = ftemp_arrh(jnp.array(c["inputs"]["tk"]), jnp.array(c["inputs"]["dha"]))
    assert jnp.allclose(result, c["output"], rtol=RTOL)


@pytest.mark.parametrize("c", PHYDRO["gammastar"],
    ids=lambda c: f"tc={c['inputs']['tc']:.1f}_patm={c['inputs']['patm']:.0f}")
def test_gammastar(c):
    result = gammastar(jnp.array(c["inputs"]["tc"]), jnp.array(c["inputs"]["patm"]))
    assert jnp.allclose(result, c["output"], rtol=RTOL)


@pytest.mark.parametrize("c", PHYDRO["ftemp_kphio_c3"],
    ids=lambda c: f"tc={c['inputs']['tc']:.1f}")
def test_ftemp_kphio_c3(c):
    result = ftemp_kphio(jnp.array(c["inputs"]["tc"]), False)
    assert jnp.allclose(result, c["output"], rtol=RTOL, atol=1e-15)


@pytest.mark.parametrize("c", PHYDRO["ftemp_kphio_c4"],
    ids=lambda c: f"tc={c['inputs']['tc']:.1f}")
def test_ftemp_kphio_c4(c):
    result = ftemp_kphio(jnp.array(c["inputs"]["tc"]), True)
    assert jnp.allclose(result, c["output"], rtol=RTOL, atol=1e-15)


@pytest.mark.parametrize("c", PHYDRO["quadratic"],
    ids=lambda c: f"a={c['inputs']['a']}_b={c['inputs']['b']}_c={c['inputs']['c']}")
def test_quadratic(c):
    inp = c["inputs"]
    result = quadratic(jnp.array(inp["a"]), jnp.array(inp["b"]), jnp.array(inp["c"]))
    assert jnp.allclose(result, c["output"], rtol=RTOL, atol=1e-15)


@pytest.mark.parametrize("c", PHYDRO["density_h2o"],
    ids=lambda c: f"tc={c['inputs']['tc']:.1f}_patm={c['inputs']['patm']:.0f}")
def test_density_h2o(c):
    result = density_h2o(jnp.array(c["inputs"]["tc"]), jnp.array(c["inputs"]["patm"]))
    assert jnp.allclose(result, c["output"], rtol=RTOL)


@pytest.mark.parametrize("c", PHYDRO["viscosity_h2o"],
    ids=lambda c: f"tc={c['inputs']['tc']:.1f}_patm={c['inputs']['patm']:.0f}")
def test_viscosity_h2o(c):
    result = viscosity_h2o(jnp.array(c["inputs"]["tc"]), jnp.array(c["inputs"]["patm"]))
    assert jnp.allclose(result, c["output"], rtol=RTOL)


@pytest.mark.parametrize("c", PHYDRO["calc_kmm"],
    ids=lambda c: f"tc={c['inputs']['tc']:.1f}_patm={c['inputs']['patm']:.0f}")
def test_calc_kmm(c):
    result = calc_kmm(jnp.array(c["inputs"]["tc"]), jnp.array(c["inputs"]["patm"]))
    assert jnp.allclose(result, c["output"], rtol=RTOL)


# ── Functions needing ParEnv ──────────────────────────────────────────


def _make_par_env(inp):
    tc = inp["tc"]
    patm = inp["patm"]
    return ParEnv(
        viscosity_water=viscosity_h2o(jnp.array(tc), jnp.array(patm)),
        density_water=density_h2o(jnp.array(tc), jnp.array(patm)),
        patm=jnp.array(patm),
        tc=jnp.array(tc),
        vpd=jnp.array(inp.get("vpd", 1000.0)),
    )


@pytest.mark.parametrize("c", PHYDRO["scale_conductivity"],
    ids=lambda c: f"tc={c['inputs']['tc']:.1f}_patm={c['inputs']['patm']:.0f}")
def test_scale_conductivity(c):
    par_env = _make_par_env(c["inputs"])
    result = scale_conductivity(jnp.array(c["inputs"]["K"]), par_env)
    assert jnp.allclose(result, c["output"], rtol=RTOL)


@pytest.mark.parametrize("c", PHYDRO["calc_gs"],
    ids=lambda c: f"tc={c['inputs']['tc']:.1f}_patm={c['inputs']['patm']:.0f}")
def test_calc_gs(c):
    inp = c["inputs"]
    par_env = _make_par_env(inp)
    par_plant = ParPlant(
        conductivity=jnp.array(inp["conductivity"]),
        psi50=jnp.array(inp["psi50"]),
        b=jnp.array(inp["b"]),
    )
    result = calc_gs(jnp.array(inp["dpsi"]), jnp.array(inp["psi_soil"]), par_plant, par_env)
    assert jnp.allclose(result, c["output"], rtol=RTOL)


@pytest.mark.parametrize("c", PHYDRO["calc_assim_light_limited"],
    ids=lambda c: f"tc={c['inputs']['tc']:.1f}_patm={c['inputs']['patm']:.0f}")
def test_calc_assim_light_limited(c):
    inp = c["inputs"]
    tc = inp["tc"]
    patm = inp["patm"]
    par_env = _make_par_env({"tc": tc, "patm": patm, "vpd": 1000.0})
    par_plant = ParPlant(
        conductivity=jnp.array(4e-16),
        psi50=jnp.array(-3.46),
        b=jnp.array(2.0),
    )
    gs = calc_gs(jnp.array(1.0), jnp.array(-0.5), par_plant, par_env)
    par_ps = ParPhotosynth(
        kmm=calc_kmm(jnp.array(tc), jnp.array(patm)),
        gammastar=gammastar(jnp.array(tc), jnp.array(patm)),
        phi0=jnp.array(KPHIO) * ftemp_kphio(jnp.array(tc), False),
        Iabs=jnp.array(inp["Iabs"]),
        ca=jnp.array(inp["ca_ppm"] * patm * 1e-6),
        patm=jnp.array(patm),
        delta=jnp.array(inp["delta"]),
    )
    ci, aj = calc_assim_light_limited(gs, jnp.array(inp["jmax"]), par_ps)
    expected = c["output"]
    assert jnp.allclose(ci, expected["ci"], rtol=RTOL, atol=1e-12)
    assert jnp.allclose(aj, expected["aj"], rtol=RTOL, atol=1e-12)


@pytest.mark.parametrize("c", PHYDRO["fn_profit"],
    ids=lambda c: f"tc={c['inputs']['tc']:.1f}_patm={c['inputs']['patm']:.0f}_{c['inputs'].get('hypothesis', 'PM')}_optim={c['inputs'].get('do_optim', False)}")
def test_fn_profit(c):
    inp = c["inputs"]
    tc = inp["tc"]
    patm = inp["patm"]
    par_env = _make_par_env({"tc": tc, "patm": patm, "vpd": inp["vpd"]})
    par_plant = ParPlant(
        conductivity=jnp.array(inp["conductivity"]),
        psi50=jnp.array(inp["psi50"]),
        b=jnp.array(inp["b"]),
    )
    par_cost = ParCost(alpha=jnp.array(inp["alpha"]), gamma=jnp.array(inp["gamma"]))
    par_ps = ParPhotosynth(
        kmm=calc_kmm(jnp.array(tc), jnp.array(patm)),
        gammastar=gammastar(jnp.array(tc), jnp.array(patm)),
        phi0=jnp.array(KPHIO) * ftemp_kphio(jnp.array(tc), False),
        Iabs=jnp.array(inp["Iabs"]),
        ca=jnp.array(inp["ca_ppm"] * patm * 1e-6),
        patm=jnp.array(patm),
        delta=jnp.array(inp["delta"]),
    )
    result = fn_profit(
        jnp.array(inp["logjmax"]), jnp.array(inp["dpsi"]),
        psi_soil=jnp.array(inp["psi_soil"]),
        par_cost=par_cost, par_photosynth=par_ps,
        par_plant=par_plant, par_env=par_env,
        hypothesis=inp.get("hypothesis", "PM"),
        do_optim=inp.get("do_optim", False),
    )
    assert jnp.allclose(result, c["output"], rtol=RTOL, atol=1e-12)


# ── Differentiability ─────────────────────────────────────────────────


def test_fn_profit_differentiable():
    """fn_profit gradient should be finite."""
    par_env = _make_par_env({"tc": 20.0, "patm": 101325.0, "vpd": 1000.0})
    par_plant = ParPlant(
        conductivity=jnp.array(4e-16), psi50=jnp.array(-3.46), b=jnp.array(2.0))
    par_cost = ParCost(alpha=jnp.array(0.1), gamma=jnp.array(0.5))
    par_ps = ParPhotosynth(
        kmm=calc_kmm(jnp.array(20.0), jnp.array(101325.0)),
        gammastar=gammastar(jnp.array(20.0), jnp.array(101325.0)),
        phi0=jnp.array(KPHIO) * ftemp_kphio(jnp.array(20.0), False),
        Iabs=jnp.array(300.0),
        ca=jnp.array(400.0 * 101325.0 * 1e-6),
        patm=jnp.array(101325.0),
        delta=jnp.array(0.015),
    )

    def profit_fn(log_jmax, dpsi):
        return fn_profit(log_jmax, dpsi, psi_soil=jnp.array(-0.5),
                         par_cost=par_cost, par_photosynth=par_ps,
                         par_plant=par_plant, par_env=par_env)

    g = jax.grad(profit_fn, argnums=(0, 1))(jnp.array(3.0), jnp.array(1.0))
    assert jnp.isfinite(g[0])
    assert jnp.isfinite(g[1])


# ── Invariant / Metamorphic Tests ───────────────────────────────────


def test_ftemp_arrh_monotone_in_temperature():
    """Arrhenius function should increase monotonically with temperature."""
    tks = jnp.linspace(260.0, 320.0, 50)
    dha = jnp.array(37830.0)
    vals = jax.vmap(lambda tk: ftemp_arrh(tk, dha))(tks)
    diffs = jnp.diff(vals)
    assert jnp.all(diffs > 0), "ftemp_arrh should increase with temperature"


def test_ftemp_arrh_unity_at_reference():
    """ftemp_arrh should equal 1.0 at the reference temperature (298.15 K)."""
    result = ftemp_arrh(jnp.array(298.15), jnp.array(50000.0))
    assert jnp.allclose(result, 1.0, atol=1e-12)


def test_gammastar_increases_with_temperature():
    """Γ* should increase with temperature (photorespiration increases)."""
    tcs = jnp.linspace(0.0, 40.0, 20)
    patm = jnp.array(101325.0)
    vals = jax.vmap(lambda tc: gammastar(tc, patm))(tcs)
    diffs = jnp.diff(vals)
    assert jnp.all(diffs > 0)


def test_gammastar_scales_with_pressure():
    """Γ* should scale linearly with atmospheric pressure."""
    tc = jnp.array(25.0)
    g1 = gammastar(tc, jnp.array(101325.0))
    g2 = gammastar(tc, jnp.array(2 * 101325.0))
    assert jnp.allclose(g2 / g1, 2.0, rtol=1e-10)


def test_density_h2o_positive():
    """Water density should be positive for all physical temperatures."""
    tcs = jnp.linspace(-10.0, 40.0, 20)
    patm = jnp.array(101325.0)
    vals = jax.vmap(lambda tc: density_h2o(tc, patm))(tcs)
    assert jnp.all(vals > 900.0)
    assert jnp.all(vals < 1100.0)


def test_viscosity_h2o_decreases_with_temperature():
    """Viscosity should decrease with temperature (molecules move faster)."""
    tcs = jnp.linspace(0.0, 40.0, 20)
    patm = jnp.array(101325.0)
    vals = jax.vmap(lambda tc: viscosity_h2o(tc, patm))(tcs)
    diffs = jnp.diff(vals)
    assert jnp.all(diffs < 0)


def test_quadratic_solves_correctly():
    """Quadratic root should satisfy the original equation (substitution check)."""
    a, b, c = jnp.array(-1.0), jnp.array(5.0), jnp.array(-6.0)
    r = quadratic(a, b, c)
    residual = a * r ** 2 + b * r + c
    assert jnp.allclose(residual, 0.0, atol=1e-12)


def test_quadratic_differentiable():
    """Gradient through quadratic should be finite (needed for backprop)."""
    g = jax.grad(lambda b: quadratic(jnp.array(-1.0), b, jnp.array(-6.0)))(jnp.array(5.0))
    assert jnp.isfinite(g)


def test_calc_kmm_gradient_finite():
    """calc_kmm gradient w.r.t. temperature should be finite."""
    g = jax.grad(lambda tc: calc_kmm(tc, jnp.array(101325.0)))(jnp.array(25.0))
    assert jnp.isfinite(g)


def test_fn_profit_ood_gradients():
    """fn_profit gradients should be finite at out-of-distribution inputs."""
    par_env = _make_par_env({"tc": 45.0, "patm": 60000.0, "vpd": 5000.0})
    par_plant = ParPlant(
        conductivity=jnp.array(4e-16), psi50=jnp.array(-3.46), b=jnp.array(2.0))
    par_cost = ParCost(alpha=jnp.array(0.1), gamma=jnp.array(0.5))
    par_ps = ParPhotosynth(
        kmm=calc_kmm(jnp.array(45.0), jnp.array(60000.0)),
        gammastar=gammastar(jnp.array(45.0), jnp.array(60000.0)),
        phi0=jnp.array(KPHIO) * ftemp_kphio(jnp.array(45.0), False),
        Iabs=jnp.array(50.0),
        ca=jnp.array(800.0 * 60000.0 * 1e-6),
        patm=jnp.array(60000.0),
        delta=jnp.array(0.015),
    )

    def profit_fn(log_jmax, dpsi):
        return fn_profit(log_jmax, dpsi, psi_soil=jnp.array(-2.0),
                         par_cost=par_cost, par_photosynth=par_ps,
                         par_plant=par_plant, par_env=par_env)

    g = jax.grad(profit_fn, argnums=(0, 1))(jnp.array(5.0), jnp.array(3.0))
    assert jnp.isfinite(g[0]), "fn_profit grad wrt log_jmax should be finite at OOD"
    assert jnp.isfinite(g[1]), "fn_profit grad wrt dpsi should be finite at OOD"
