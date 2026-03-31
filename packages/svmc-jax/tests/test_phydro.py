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


# ── pmodel_hydraulics_numerical (full solver) ──────────────────────────


from svmc_jax.phydro.solver import pmodel_hydraulics_numerical


@pytest.mark.parametrize("c", PHYDRO.get("pmodel_hydraulics_numerical", []),
    ids=lambda c: (
        f"tc={c['inputs']['tc']}_vpd={c['inputs']['vpd']}"
        f"_psi={c['inputs']['psi_soil']}"
    ))
def test_pmodel_hydraulics_numerical(c):
    """Solver output must match Fortran reference within optimizer tolerance."""
    inp = c["inputs"]
    result = pmodel_hydraulics_numerical(
        tc=inp["tc"],
        ppfd=inp["ppfd"],
        vpd=inp["vpd"],
        co2=inp["co2"],
        sp=inp["sp"],
        fapar=inp["fapar"],
        psi_soil=inp["psi_soil"],
        rdark_leaf=inp["rdark_leaf"],
    )
    ref = c["output"]
    # Optimizer convergence tolerance: JAX uses exact autodiff gradients while
    # Fortran uses finite-difference (h=0.001), so they converge to slightly
    # different optima. Observed relative differences: jmax ~3-7e-4, dpsi ~1-3e-3.
    rtol_solver = 3e-3
    for key in ("jmax", "dpsi", "gs", "aj", "ci", "chi", "vcmax", "profit"):
        assert jnp.allclose(result[key], ref[key], rtol=rtol_solver, atol=1e-10), \
            f"{key}: JAX={float(result[key]):.10g} vs Fortran={ref[key]:.10g}"


def test_pmodel_hydraulics_numerical_profit_positive():
    """At benign conditions, optimised profit should be positive."""
    result = pmodel_hydraulics_numerical(
        tc=20.0, ppfd=300.0, vpd=1000.0, co2=400.0,
        sp=101325.0, fapar=0.9, psi_soil=-0.5, rdark_leaf=0.015,
    )
    assert float(result["profit"]) > 0.0
    assert float(result["jmax"]) > 0.0
    assert float(result["gs"]) > 0.0


def test_pmodel_hydraulics_numerical_differentiable():
    """Gradients through the full solver should be finite."""

    def loss(psi_soil):
        result = pmodel_hydraulics_numerical(
            tc=20.0, ppfd=300.0, vpd=1000.0, co2=400.0,
            sp=101325.0, fapar=0.9, psi_soil=psi_soil, rdark_leaf=0.015,
        )
        return result["gs"]

    g = jax.grad(loss)(jnp.array(-0.5))
    assert jnp.isfinite(g), f"Gradient through full solver is not finite: {g}"


def test_pmodel_hydraulics_numerical_jit_consistent():
    """Direct and jitted solver calls should agree bitwise enough."""
    kwargs = dict(
        tc=20.0, ppfd=300.0, vpd=1000.0, co2=400.0,
        sp=101325.0, fapar=0.9, psi_soil=-0.5, rdark_leaf=0.015,
    )
    direct = pmodel_hydraulics_numerical(**kwargs)
    jitted = jax.jit(lambda: pmodel_hydraulics_numerical(**kwargs))()

    for key in ("jmax", "dpsi", "gs", "aj", "ci", "chi", "vcmax", "profit"):
        assert jnp.allclose(direct[key], jitted[key], rtol=5e-8, atol=1e-12), \
            f"{key}: direct={float(direct[key]):.10g} vs jit={float(jitted[key]):.10g}"


def test_pmodel_hydraulics_numerical_projected_newton_selectable():
    """The committed projected-Newton baseline should remain callable."""
    result = pmodel_hydraulics_numerical(
        tc=20.0, ppfd=300.0, vpd=1000.0, co2=400.0,
        sp=101325.0, fapar=0.9, psi_soil=-0.5, rdark_leaf=0.015,
        solver_kind="projected_newton",
    )

    for key in ("jmax", "dpsi", "gs", "aj", "ci", "chi", "vcmax", "profit"):
        assert jnp.isfinite(result[key]), f"{key} is not finite for projected_newton"


def test_solver_monotonic_vpd():
    """Higher VPD should reduce stomatal conductance (gs) — basic ecophysiology."""
    vpds = [500.0, 1000.0, 2000.0, 3000.0]
    gs_values = []
    for v in vpds:
        r = pmodel_hydraulics_numerical(
            tc=20.0, ppfd=300.0, vpd=v, co2=400.0,
            sp=101325.0, fapar=0.9, psi_soil=-0.5, rdark_leaf=0.015,
        )
        gs_values.append(float(r["gs"]))
    for i in range(len(gs_values) - 1):
        assert gs_values[i] > gs_values[i + 1], \
            f"gs should decrease with VPD: gs({vpds[i]})={gs_values[i]} >= gs({vpds[i+1]})={gs_values[i+1]}"


def test_solver_monotonic_drought():
    """Drier soil (lower ψ_soil) should reduce stomatal conductance."""
    psi_soils = [-0.2, -0.5, -1.0, -2.0]
    gs_values = []
    for ps in psi_soils:
        r = pmodel_hydraulics_numerical(
            tc=20.0, ppfd=300.0, vpd=1000.0, co2=400.0,
            sp=101325.0, fapar=0.9, psi_soil=ps, rdark_leaf=0.015,
        )
        gs_values.append(float(r["gs"]))
    for i in range(len(gs_values) - 1):
        assert gs_values[i] > gs_values[i + 1], \
            f"gs should decrease with drought: gs({psi_soils[i]})={gs_values[i]} >= gs({psi_soils[i+1]})={gs_values[i+1]}"


def test_solver_outputs_consistent():
    """Verify internal consistency: aj = gs0 * (ca - ci)."""
    r = pmodel_hydraulics_numerical(
        tc=20.0, ppfd=300.0, vpd=1000.0, co2=400.0,
        sp=101325.0, fapar=0.9, psi_soil=-0.5, rdark_leaf=0.015,
    )
    gs0 = float(r["gs"]) * 1e6 / 101325.0
    ca = 400.0 * 101325.0 * 1e-6
    aj_recomputed = gs0 * (ca - float(r["ci"]))
    assert jnp.allclose(jnp.float64(aj_recomputed), r["aj"], rtol=1e-8), \
        f"aj consistency: {aj_recomputed} vs {float(r['aj'])}"


# ── Solver comparison test set ──────────────────────────────────────────
#
# Compare projected-Newton against scipy L-BFGS-B on random plausible
# environments, plus gradient checks.  Catches off-fixture solver failures
# and custom_vjp inconsistencies.

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from svmc_jax.phydro.solver import (
    _objective,
    _LOG_JMAX_LO,
    _LOG_JMAX_HI,
    _DPSI_LO,
    _DPSI_HI,
)


def _build_params(tc, ppfd, vpd, co2=400.0, sp=101325.0, fapar=0.9,
                  psi_soil=-0.5, rdark_leaf=0.015, alpha=0.1,
                  gamma_cost=0.5, conductivity=4e-16, psi50=-3.46,
                  b_param=2.0, kphio=KPHIO):
    """Build parameter structs from scalar inputs for objective evaluation."""
    from svmc_jax.phydro.leaf_functions import (
        scale_conductivity, ParPhotosynth, ParPlant, ParCost, ParEnv,
    )
    tc_ = jnp.float64(tc)
    sp_ = jnp.float64(sp)
    vpd_ = jnp.float64(vpd)
    ppfd_ = jnp.float64(ppfd)
    co2_ = jnp.float64(co2)
    par_plant = ParPlant(
        conductivity=jnp.float64(conductivity),
        psi50=jnp.float64(psi50),
        b=jnp.float64(b_param),
    )
    par_cost = ParCost(alpha=jnp.float64(alpha), gamma=jnp.float64(gamma_cost))
    kmm = calc_kmm(tc_, sp_)
    gs_val = gammastar(tc_, sp_)
    phi0 = jnp.float64(kphio) * ftemp_kphio(tc_, c4=False)
    Iabs = ppfd_ * jnp.float64(fapar)
    ca = co2_ * sp_ * 1e-6
    par_photosynth = ParPhotosynth(
        kmm=kmm, gammastar=gs_val, phi0=phi0,
        Iabs=Iabs, ca=ca, patm=sp_, delta=jnp.float64(rdark_leaf),
    )
    par_env = ParEnv(
        viscosity_water=viscosity_h2o(tc_, sp_),
        density_water=density_h2o(tc_, sp_),
        patm=sp_, tc=tc_, vpd=vpd_,
    )
    return jnp.float64(psi_soil), par_cost, par_photosynth, par_plant, par_env


def _scipy_reference(psi_soil, par_cost, par_photosynth, par_plant, par_env):
    """Solve with scipy L-BFGS-B as ground-truth reference."""
    def obj_np(x):
        val = _objective(
            jnp.array(x), psi_soil, par_cost, par_photosynth,
            par_plant, par_env,
        )
        return float(val)

    bounds = [(_LOG_JMAX_LO, _LOG_JMAX_HI), (_DPSI_LO, _DPSI_HI)]
    res = scipy_minimize(obj_np, [4.0, 1.0], method="L-BFGS-B", bounds=bounds)
    return res.x, res.fun


def _assert_solver_matches_scipy(case, gap_tol):
    """Blocking regression helper for known solver failure pockets."""
    params = _build_params(**case)
    newton_result = pmodel_hydraulics_numerical(
        tc=case["tc"],
        ppfd=case["ppfd"],
        vpd=case["vpd"],
        co2=case.get("co2", 400.0),
        sp=case.get("sp", 101325.0),
        fapar=case.get("fapar", 1.0),
        psi_soil=case["psi_soil"],
        rdark_leaf=case.get("rdark_leaf", 0.015),
        conductivity=case.get("conductivity", 4e-16),
        psi50=case.get("psi50", -3.46),
        b_param=case.get("b_param", 2.0),
        alpha=case["alpha"],
        gamma_cost=case.get("gamma_cost", 0.5),
        kphio=case.get("kphio", KPHIO),
    )
    log_jmax = float(jnp.log(newton_result["jmax"]))
    dpsi = float(newton_result["dpsi"])
    assert jnp.isfinite(newton_result["jmax"]), "solver returned non-finite jmax"
    assert jnp.isfinite(newton_result["dpsi"]), "solver returned non-finite dpsi"
    newton_x = jnp.array([log_jmax, dpsi])
    newton_obj = float(_objective(newton_x, *params))

    scipy_x, scipy_obj = _scipy_reference(*params)
    gap = newton_obj - scipy_obj
    assert gap < gap_tol, (
        f"Newton x={[log_jmax, dpsi]} obj={newton_obj:.12g} vs "
        f"scipy x={scipy_x.tolist()} obj={scipy_obj:.12g}; gap={gap:.12g}"
    )


# 50 random plausible environments with fixed seed for reproducibility
_RNG = np.random.default_rng(42)
_N_RANDOM = 50
_RANDOM_CASES = [
    {
        "tc": float(_RNG.uniform(0, 35)),
        "ppfd": float(_RNG.uniform(50, 2000)),
        "vpd": float(_RNG.uniform(100, 4000)),
        "psi_soil": float(_RNG.uniform(-3.0, -0.1)),
        "alpha": float(_RNG.uniform(0.05, 0.3)),
    }
    for _ in range(_N_RANDOM)
]


@pytest.mark.parametrize(
    "case", _RANDOM_CASES,
    ids=lambda c: f"tc={c['tc']:.1f}_ppfd={c['ppfd']:.0f}_vpd={c['vpd']:.0f}_psi={c['psi_soil']:.2f}",
)
def test_solver_vs_scipy(case):
    """Newton solver must reach objective within 0.5 of scipy L-BFGS-B."""
    params = _build_params(**case)
    # Newton solution
    newton_result = pmodel_hydraulics_numerical(
        tc=case["tc"], ppfd=case["ppfd"], vpd=case["vpd"],
        co2=400.0, sp=101325.0, fapar=0.9,
        psi_soil=case["psi_soil"], rdark_leaf=0.015,
        alpha=case["alpha"],
    )
    newton_x = jnp.array([jnp.log(newton_result["jmax"]),
                           newton_result["dpsi"]])
    newton_obj = float(_objective(newton_x, *params))
    # Scipy reference
    _, scipy_obj = _scipy_reference(*params)
    gap = newton_obj - scipy_obj  # positive = Newton is worse
    assert gap < 0.5, (
        f"Newton obj={newton_obj:.4f} vs scipy obj={scipy_obj:.4f}, "
        f"gap={gap:.4f} exceeds 0.5"
    )


# Specific regression case: previously stalled at initial guess
def test_solver_regression_stall_case():
    """Regression: the case that stalled with fixed-regularisation Newton."""
    r = pmodel_hydraulics_numerical(
        tc=4.5446371446088065, ppfd=1344.5858935751426,
        vpd=2319.557259496112, co2=400.0, sp=101325.0,
        fapar=0.9, psi_soil=-0.3208819059598719, rdark_leaf=0.015,
    )
    # Must not return initial guess (log_jmax=4.0, dpsi=1.0)
    log_jmax = float(jnp.log(r["jmax"]))
    dpsi = float(r["dpsi"])
    assert abs(log_jmax - 4.0) > 0.1 or abs(dpsi - 1.0) > 0.1, \
        f"Solver stalled at initial guess: log_jmax={log_jmax}, dpsi={dpsi}"
    assert float(r["profit"]) > 5.0, \
        f"Profit too low ({float(r['profit']):.2f}), solver likely stalled"


def test_solver_regression_low_light_high_cost_hits_lower_bound_optimum():
    """Blocking regression: low-light/high-cost cases should reach scipy optimum."""
    _assert_solver_matches_scipy(
        dict(
            tc=-5.0,
            ppfd=100.0,
            vpd=1000.0,
            psi_soil=-1.0,
            alpha=5.0,
            gamma_cost=0.5,
            co2=400.0,
            sp=101325.0,
            fapar=1.0,
            rdark_leaf=0.015,
        ),
        gap_tol=1e-4,
    )


def test_solver_regression_high_vpd_interior_case_matches_scipy():
    """Blocking regression: interior high-VPD optimum should not under-converge."""
    _assert_solver_matches_scipy(
        dict(
            tc=15.0,
            ppfd=2200.0,
            vpd=4500.0,
            psi_soil=-0.5,
            alpha=0.03,
            gamma_cost=0.2,
            co2=400.0,
            sp=101325.0,
            fapar=1.0,
            rdark_leaf=0.015,
        ),
        gap_tol=1e-3,
    )


def test_solver_regression_weak_hydraulics_case_matches_scipy():
    """Blocking regression: weak-hydraulics dry-soil optimum should match scipy."""
    _assert_solver_matches_scipy(
        dict(
            tc=25.0,
            ppfd=1000.0,
            vpd=800.0,
            psi_soil=-4.0,
            alpha=0.05,
            gamma_cost=0.5,
            conductivity=1e-17,
            psi50=-2.0,
            b_param=4.0,
            co2=400.0,
            sp=101325.0,
            fapar=1.0,
            rdark_leaf=0.015,
        ),
        gap_tol=1e-6,
    )


def test_solver_regression_random_nan_case_matches_scipy():
    """Blocking regression: previous NaN case should stay finite and optimal."""
    _assert_solver_matches_scipy(
        dict(
            tc=31.797319881432422,
            ppfd=1151.314009508769,
            vpd=625.4710276386504,
            psi_soil=-3.6668503071422798,
            alpha=0.013313615865987496,
            gamma_cost=0.45378309233645575,
            conductivity=2.2352797950193703e-17,
            psi50=-1.420224220107091,
            b_param=5.506985418171395,
            co2=400.0,
            sp=101325.0,
            fapar=1.0,
            rdark_leaf=0.015,
        ),
        gap_tol=1e-6,
    )


def test_solver_regression_lower_bound_start_case_matches_scipy():
    """Blocking regression: bad interior start should be rescued by fixed multistart."""
    _assert_solver_matches_scipy(
        dict(
            tc=15.820147089607975,
            ppfd=1723.6724451871746,
            vpd=1608.8633604073505,
            psi_soil=-4.800500207103255,
            alpha=4.174344486657861,
            gamma_cost=1.031605162555923,
            conductivity=3.078206259265897e-17,
            psi50=-1.3587134297601686,
            b_param=4.675975262634204,
            co2=400.0,
            sp=101325.0,
            fapar=1.0,
            rdark_leaf=0.015,
        ),
        gap_tol=1e-6,
    )


# Gradient checks: AD vs finite differences
_GRAD_CASES = [
    # Benign interior case
    {"tc": 20.0, "ppfd": 300.0, "vpd": 1000.0, "psi_soil": -0.5,
    "alpha": 0.1, "label": "benign"},
    # Former stall case
    {"tc": 4.54, "ppfd": 1344.6, "vpd": 2319.6, "psi_soil": -0.321,
    "alpha": 0.1, "label": "former_stall"},
    # Dry + high VPD (bound-active region)
    {"tc": 30.0, "ppfd": 800.0, "vpd": 3500.0, "psi_soil": -2.5,
    "alpha": 0.1, "label": "dry_high_vpd"},
    # Cold + low light
    {"tc": 2.0, "ppfd": 80.0, "vpd": 200.0, "psi_soil": -0.3,
    "alpha": 0.1, "label": "cold_low_light"},
    # Exact lower-bound optimum for dpsi: constrained VJP must zero it out.
    {"tc": 20.0, "ppfd": 300.0, "vpd": 1000.0, "psi_soil": -0.5,
    "alpha": 5.0, "label": "very_costly_bound"},
]


@pytest.mark.parametrize("case", _GRAD_CASES, ids=lambda c: c["label"])
def test_solver_gradient_vs_finite_diff(case):
    """AD gradient through full solver must agree with finite differences."""
    def loss(alpha):
        return pmodel_hydraulics_numerical(
            tc=case["tc"], ppfd=case["ppfd"], vpd=case["vpd"],
            co2=400.0, sp=101325.0, fapar=0.9,
            psi_soil=case["psi_soil"], rdark_leaf=0.015,
            alpha=alpha,
        )["aj"]

    alpha0 = case["alpha"]
    g_ad = float(jax.grad(loss)(alpha0))

    # Central finite differences at eps=1e-4
    eps = 1e-4
    g_fd = float((loss(alpha0 + eps) - loss(alpha0 - eps)) / (2 * eps))

    # Check agreement (allow 1% relative or 0.1 absolute for near-zero grads)
    if abs(g_fd) > 0.1:
        rel = abs(g_ad - g_fd) / abs(g_fd)
        assert rel < 0.01, (
            f"[{case['label']}] AD={g_ad:.6e} vs FD={g_fd:.6e}, rel={rel:.3e}"
        )
    else:
        assert abs(g_ad - g_fd) < 0.1, (
            f"[{case['label']}] AD={g_ad:.6e} vs FD={g_fd:.6e}, "
            f"abs diff={abs(g_ad - g_fd):.3e}"
        )
