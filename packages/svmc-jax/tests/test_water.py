"""Tests for SpaFHy water module leaf functions in JAX — validated against Fortran reference."""

import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from svmc_jax.water.leaf_functions import (
    e_sat,
    penman_monteith,
    soil_water_retention_curve,
    soil_hydraulic_conductivity,
    aerodynamics,
    exponential_smooth_met,
    SoilHydroParams,
    AeroParams,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "../../svmc-ref/fixtures"
WATER = json.loads((FIXTURES_DIR / "water.json").read_text())

RTOL = 1e-10


# ── e_sat ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("c", WATER["e_sat"],
    ids=lambda c: f"tc={c['inputs']['tc']:.1f}_patm={c['inputs']['patm']:.0f}")
def test_e_sat(c):
    inp = c["inputs"]
    esat, s, gamma = e_sat(jnp.array(inp["tc"]), jnp.array(inp["patm"]))
    exp = c["output"]
    assert jnp.allclose(esat, exp["esat"], rtol=RTOL)
    assert jnp.allclose(s, exp["s"], rtol=RTOL)
    assert jnp.allclose(gamma, exp["g"], rtol=RTOL)


# ── penman_monteith ───────────────────────────────────────────────────


@pytest.mark.parametrize("c", WATER["penman_monteith"],
    ids=lambda c: f"tc={c['inputs']['tc']:.1f}_patm={c['inputs']['patm']:.0f}_vpd={c['inputs']['vpd']:.0f}")
def test_penman_monteith(c):
    inp = c["inputs"]
    result = penman_monteith(
        ae=jnp.array(inp["AE"]),
        d=jnp.array(inp["vpd"]),
        tc=jnp.array(inp["tc"]),
        gs=jnp.array(inp["Gs"]),
        ga=jnp.array(inp["Ga"]),
        patm=jnp.array(inp["patm"]),
    )
    assert jnp.allclose(result, c["output"], rtol=RTOL)


# ── soil_water_retention_curve ────────────────────────────────────────


@pytest.mark.parametrize("c", WATER["soil_water_retention_curve"],
    ids=lambda c: f"vol_liq={c['inputs']['vol_liq']:.2f}")
def test_soil_water_retention_curve(c):
    inp = c["inputs"]
    params = SoilHydroParams(
        watsat=jnp.array(inp["watsat"]),
        watres=jnp.array(inp["watres"]),
        alpha_van=jnp.array(inp["alpha_van"]),
        n_van=jnp.array(inp["n_van"]),
        ksat=jnp.array(1e-5),
    )
    smp = soil_water_retention_curve(jnp.array(inp["vol_liq"]), params)
    assert jnp.allclose(smp, c["output"], rtol=RTOL)


# ── soil_hydraulic_conductivity ───────────────────────────────────────


@pytest.mark.parametrize("c", WATER["soil_hydraulic_conductivity"],
    ids=lambda c: f"vol_liq={c['inputs']['vol_liq']:.2f}")
def test_soil_hydraulic_conductivity(c):
    inp = c["inputs"]
    params = SoilHydroParams(
        watsat=jnp.array(inp["watsat"]),
        watres=jnp.array(inp["watres"]),
        alpha_van=jnp.array(inp["alpha_van"]),
        n_van=jnp.array(inp["n_van"]),
        ksat=jnp.array(inp["ksat"]),
    )
    k = soil_hydraulic_conductivity(jnp.array(inp["vol_liq"]), params)
    assert jnp.allclose(k, c["output"], rtol=RTOL, atol=1e-30)


# ── aerodynamics ──────────────────────────────────────────────────────


@pytest.mark.parametrize("c", WATER.get("aerodynamics", []),
    ids=lambda c: f"LAI={c['inputs']['LAI']}_Uo={c['inputs']['Uo']}")
def test_aerodynamics(c):
    inp = c["inputs"]
    params = AeroParams(
        hc=jnp.array(inp["hc"]),
        zmeas=jnp.array(inp["zmeas"]),
        zground=jnp.array(inp["zground"]),
        zo_ground=jnp.array(inp["zo_ground"]),
        w_leaf=jnp.array(inp["w_leaf"]),
    )
    ra, rb, ras, ustar, Uh, Ug = aerodynamics(jnp.array(inp["LAI"]), jnp.array(inp["Uo"]), params)
    exp = c["output"]
    assert jnp.allclose(ra, exp["ra"], rtol=RTOL)
    assert jnp.allclose(rb, exp["rb"], rtol=RTOL)
    assert jnp.allclose(ras, exp["ras"], rtol=RTOL)
    assert jnp.allclose(ustar, exp["ustar"], rtol=RTOL)
    assert jnp.allclose(Uh, exp["Uh"], rtol=RTOL)
    assert jnp.allclose(Ug, exp["Ug"], rtol=RTOL)


def test_aerodynamics_positive():
    params = AeroParams(
        hc=jnp.array(15.0), zmeas=jnp.array(2.0),
        zground=jnp.array(0.5), zo_ground=jnp.array(0.01),
        w_leaf=jnp.array(0.02))
    ra, rb, ras, ustar, Uh, Ug = aerodynamics(jnp.array(3.0), jnp.array(3.0), params)
    assert float(ra) > 0
    assert float(ras) > 0
    assert float(ustar) > 0
    assert float(Uh) > 0
    assert float(Ug) > 0


# ── Differentiability ─────────────────────────────────────────────────


def test_soil_retention_differentiable():
    params = SoilHydroParams(
        watsat=jnp.array(0.75), watres=jnp.array(0.0),
        alpha_van=jnp.array(4.45), n_van=jnp.array(1.12),
        ksat=jnp.array(1e-5))
    g = jax.grad(lambda theta: soil_water_retention_curve(theta, params))(jnp.array(0.3))
    assert jnp.isfinite(g)


# ── exponential_smooth_met ────────────────────────────────────────────


@pytest.mark.parametrize("c", WATER["exponential_smooth_met"],
    ids=lambda c: f"ind={c['inputs']['met_ind_in']}")
def test_exponential_smooth_met(c):
    inp = c["inputs"]
    met_daily = jnp.array(inp["met_daily"])
    met_rolling = jnp.array(inp["met_rolling_in"])
    met_ind = inp["met_ind_in"]

    new_rolling, new_ind = exponential_smooth_met(met_daily, met_rolling, met_ind)
    exp = c["output"]
    assert jnp.allclose(new_rolling, jnp.array(exp["met_rolling"]), rtol=1e-10)
    assert new_ind == exp["met_ind"]


# ── Invariant / Metamorphic Tests ───────────────────────────────────


def test_e_sat_increases_with_temperature():
    """Saturation vapour pressure must increase with temperature (Clausius-Clapeyron)."""
    tcs = jnp.linspace(-10.0, 40.0, 20)
    patm = jnp.array(101325.0)
    esats = jax.vmap(lambda tc: e_sat(tc, patm)[0])(tcs)
    diffs = jnp.diff(esats)
    assert jnp.all(diffs > 0)


def test_penman_monteith_non_negative():
    """Latent heat flux must be non-negative (max(LE, 0) in Fortran)."""
    result = penman_monteith(
        ae=jnp.array(-100.0), d=jnp.array(100.0),
        tc=jnp.array(10.0), gs=jnp.array(0.001),
        ga=jnp.array(0.01), patm=jnp.array(101325.0))
    assert float(result) >= 0.0


def test_soil_retention_monotone():
    """Matric potential should decrease (become more negative) as soil dries."""
    params = SoilHydroParams(
        watsat=jnp.array(0.75), watres=jnp.array(0.0),
        alpha_van=jnp.array(4.45), n_van=jnp.array(1.12),
        ksat=jnp.array(1e-5))
    thetas = jnp.linspace(0.15, 0.70, 20)
    smps = jax.vmap(lambda t: soil_water_retention_curve(t, params))(thetas)
    diffs = jnp.diff(smps)
    assert jnp.all(diffs > 0), "matric potential should increase (less negative) with moisture"


def test_soil_conductivity_monotone():
    """Hydraulic conductivity should increase with soil moisture."""
    params = SoilHydroParams(
        watsat=jnp.array(0.75), watres=jnp.array(0.0),
        alpha_van=jnp.array(4.45), n_van=jnp.array(1.12),
        ksat=jnp.array(1e-5))
    thetas = jnp.linspace(0.15, 0.70, 20)
    ks = jax.vmap(lambda t: soil_hydraulic_conductivity(t, params))(thetas)
    diffs = jnp.diff(ks)
    assert jnp.all(diffs > 0)
