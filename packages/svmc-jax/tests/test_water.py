"""Tests for SpaFHy water module — validated against Fortran reference."""

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
from svmc_jax.water.canopy_soil import (
    ground_evaporation,
    canopy_water_snow,
    canopy_water_flux,
    soil_water,
    CanopyWaterState,
    CanopySnowParams,
    SoilWaterState,
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


# ══════════════════════════════════════════════════════════════════════
# Phase 3 — canopy_water_snow, ground_evaporation, canopy_water_flux,
#            soil_water
# ══════════════════════════════════════════════════════════════════════


# ── ground_evaporation ────────────────────────────────────────────────


@pytest.mark.parametrize("c", WATER["ground_evaporation"],
    ids=lambda c: f"T={c['inputs']['T']:.0f}_SWE={c['inputs']['SWE']:.1f}")
def test_ground_evaporation(c):
    inp = c["inputs"]
    result = ground_evaporation(
        tc=jnp.array(inp["T"]),
        ae=jnp.array(inp["AE"]),
        vpd=jnp.array(inp["VPD"]),
        ras=jnp.array(inp["Ras"]),
        patm=jnp.array(inp["P"]),
        swe=jnp.array(inp["SWE"]),
        beta=jnp.array(inp["beta"]),
        wat_sto=jnp.array(inp["WatSto"]),
        gsoil=jnp.array(inp["gsoil"]),
        time_step=jnp.array(inp["time_step"]),
    )
    assert jnp.allclose(result, c["output"]["SoilEvap"], rtol=RTOL)


# ── canopy_water_snow ─────────────────────────────────────────────────


@pytest.mark.parametrize("c", WATER["canopy_water_snow"],
    ids=lambda c: f"T={c['inputs']['T']:.0f}_Pre={c['inputs']['Pre']}")
def test_canopy_water_snow(c):
    inp = c["inputs"]
    state = CanopyWaterState(
        CanopyStorage=jnp.array(inp["CanopyStorage_in"]),
        SWE=jnp.array(inp["SWE_in"]),
        swe_i=jnp.array(inp["swe_i_in"]),
        swe_l=jnp.array(inp["swe_l_in"]),
    )
    params = CanopySnowParams(
        wmax=jnp.array(inp["wmax"]),
        wmaxsnow=jnp.array(inp["wmaxsnow"]),
        kmelt=jnp.array(inp["kmelt"]),
        kfreeze=jnp.array(inp["kfreeze"]),
        frac_snowliq=jnp.array(inp["frac_snowliq"]),
        gsoil=jnp.array(0.0),  # not used by canopy_water_snow
    )
    new_state, flux = canopy_water_snow(
        state, params,
        tc=jnp.array(inp["T"]),
        pre=jnp.array(inp["Pre"]),
        ae=jnp.array(inp["AE"]),
        d=jnp.array(inp["D"]),
        ra=jnp.array(inp["Ra"]),
        u=jnp.array(inp["U"]),
        lai=jnp.array(inp["LAI"]),
        patm=jnp.array(inp["P"]),
        time_step=jnp.array(inp["time_step"]),
    )
    exp = c["output"]
    # State
    assert jnp.allclose(new_state.CanopyStorage, exp["CanopyStorage"], rtol=RTOL)
    assert jnp.allclose(new_state.SWE, exp["SWE"], rtol=RTOL)
    assert jnp.allclose(new_state.swe_i, exp["swe_i"], rtol=RTOL)
    assert jnp.allclose(new_state.swe_l, exp["swe_l"], rtol=RTOL)
    # Fluxes
    assert jnp.allclose(flux.Throughfall, exp["Throughfall"], rtol=RTOL)
    assert jnp.allclose(flux.Interception, exp["Interception"], rtol=RTOL)
    assert jnp.allclose(flux.CanopyEvap, exp["CanopyEvap"], rtol=RTOL)
    assert jnp.allclose(flux.Unloading, exp["Unloading"], rtol=RTOL)
    assert jnp.allclose(flux.PotInfiltration, exp["PotInfiltration"], rtol=RTOL)
    assert jnp.allclose(flux.Melt, exp["Melt"], rtol=RTOL)
    assert jnp.allclose(flux.Freeze, exp["Freeze"], rtol=RTOL)
    assert jnp.allclose(flux.mbe, exp["mbe"], atol=1e-10)


# ── canopy_water_flux ─────────────────────────────────────────────────


@pytest.mark.parametrize("c", WATER["canopy_water_flux"],
    ids=lambda c: f"Ta={c['inputs']['Ta']:.0f}_LAI={c['inputs']['LAI']:.0f}")
def test_canopy_water_flux(c):
    inp = c["inputs"]
    cw_state = CanopyWaterState(
        CanopyStorage=jnp.array(inp["CanopyStorage_in"]),
        SWE=jnp.array(inp["SWE_in"]),
        swe_i=jnp.array(inp["swe_i_in"]),
        swe_l=jnp.array(inp["swe_l_in"]),
    )
    aero_params = AeroParams(
        hc=jnp.array(inp["hc"]),
        zmeas=jnp.array(inp["zmeas"]),
        zground=jnp.array(inp["zground"]),
        zo_ground=jnp.array(inp["zo_ground"]),
        w_leaf=jnp.array(inp["w_leaf"]),
    )
    cs_params = CanopySnowParams(
        wmax=jnp.array(inp["wmax"]),
        wmaxsnow=jnp.array(inp["wmaxsnow"]),
        kmelt=jnp.array(inp["kmelt"]),
        kfreeze=jnp.array(inp["kfreeze"]),
        frac_snowliq=jnp.array(inp["frac_snowliq"]),
        gsoil=jnp.array(inp["gsoil"]),
    )
    new_state, flux = canopy_water_flux(
        rn=jnp.array(inp["Rn"]),
        ta=jnp.array(inp["Ta"]),
        prec=jnp.array(inp["Prec"]),
        vpd=jnp.array(inp["VPD"]),
        u=jnp.array(inp["U"]),
        patm=jnp.array(inp["P"]),
        fapar=jnp.array(inp["fapar"]),
        lai=jnp.array(inp["LAI"]),
        cw_state=cw_state,
        sw_beta=jnp.array(inp["beta_in"]),
        sw_wat_sto=jnp.array(inp["WatSto_in"]),
        aero_params=aero_params,
        cs_params=cs_params,
        time_step=jnp.array(inp["time_step"]),
    )
    exp = c["output"]
    # State
    assert jnp.allclose(new_state.CanopyStorage, exp["CanopyStorage"], rtol=RTOL)
    assert jnp.allclose(new_state.SWE, exp["SWE"], rtol=RTOL)
    assert jnp.allclose(new_state.swe_i, exp["swe_i"], rtol=RTOL)
    assert jnp.allclose(new_state.swe_l, exp["swe_l"], rtol=RTOL)
    # Fluxes
    assert jnp.allclose(flux.Throughfall, exp["Throughfall"], rtol=RTOL)
    assert jnp.allclose(flux.Interception, exp["Interception"], rtol=RTOL)
    assert jnp.allclose(flux.CanopyEvap, exp["CanopyEvap"], rtol=RTOL)
    assert jnp.allclose(flux.Unloading, exp["Unloading"], rtol=RTOL)
    assert jnp.allclose(flux.SoilEvap, exp["SoilEvap"], rtol=RTOL)
    assert jnp.allclose(flux.PotInfiltration, exp["PotInfiltration"], rtol=RTOL)
    assert jnp.allclose(flux.Melt, exp["Melt"], rtol=RTOL)
    assert jnp.allclose(flux.Freeze, exp["Freeze"], rtol=RTOL)
    assert jnp.allclose(flux.mbe, exp["mbe"], atol=1e-10)


# ── soil_water ────────────────────────────────────────────────────────


@pytest.mark.parametrize("c", WATER["soil_water"],
    ids=lambda c: f"potinf={c['inputs']['potinf']:.0f}_WatSto={c['inputs']['WatSto_in']:.0f}")
def test_soil_water(c):
    inp = c["inputs"]
    state = SoilWaterState(
        WatSto=jnp.array(inp["WatSto_in"]),
        PondSto=jnp.array(inp["PondSto_in"]),
        MaxWatSto=jnp.array(inp["MaxWatSto"]),
        MaxPondSto=jnp.array(inp["MaxPondSto"]),
        FcSto=jnp.array(inp["FcSto"]),
        Wliq=jnp.array(inp["Wliq_in"]),
        Psi=jnp.array(0.0),
        Sat=jnp.array(0.0),
        Kh=jnp.array(inp["Kh_in"]),
        beta=jnp.array(0.0),
    )
    soil_params = SoilHydroParams(
        watsat=jnp.array(inp["watsat"]),
        watres=jnp.array(inp["watres"]),
        alpha_van=jnp.array(inp["alpha_van"]),
        n_van=jnp.array(inp["n_van"]),
        ksat=jnp.array(inp["ksat"]),
    )
    new_state, flux, tr_out, evap_out, latflow_out = soil_water(
        state, soil_params,
        max_poros=jnp.array(inp["max_poros"]),
        potinf=jnp.array(inp["potinf"]),
        tr=jnp.array(inp["tr"]),
        evap=jnp.array(inp["evap"]),
        latflow=jnp.array(inp["latflow"]),
        time_step=jnp.array(inp["time_step"]),
    )
    exp = c["output"]
    # State
    assert jnp.allclose(new_state.WatSto, exp["WatSto"], rtol=RTOL)
    assert jnp.allclose(new_state.PondSto, exp["PondSto"], rtol=RTOL)
    assert jnp.allclose(new_state.Wliq, exp["Wliq"], rtol=RTOL)
    assert jnp.allclose(new_state.Sat, exp["Sat"], rtol=RTOL)
    assert jnp.allclose(new_state.beta, exp["beta"], rtol=RTOL)
    assert jnp.allclose(new_state.Psi, exp["Psi"], rtol=RTOL)
    assert jnp.allclose(new_state.Kh, exp["Kh"], rtol=RTOL, atol=1e-30)
    # Fluxes
    assert jnp.allclose(flux.Infiltration, exp["Infiltration"], rtol=RTOL)
    assert jnp.allclose(flux.Drainage, exp["Drainage"], rtol=RTOL)
    assert jnp.allclose(flux.ET, exp["ET"], rtol=RTOL)
    assert jnp.allclose(flux.Runoff, exp["Runoff"], rtol=RTOL, atol=1e-15)
    assert jnp.allclose(flux.LateralFlow, exp["LateralFlow"], atol=1e-15)
    assert jnp.allclose(flux.mbe, exp["mbe"], atol=1e-10)
    # Modified inout args
    assert jnp.allclose(tr_out, exp["tr_out"], rtol=RTOL)
    assert jnp.allclose(evap_out, exp["evap_out"], rtol=RTOL)
    assert jnp.allclose(latflow_out, exp["latflow_out"], atol=1e-15)


# ══════════════════════════════════════════════════════════════════════
# Phase 3 — Invariant / Metamorphic Tests
# ══════════════════════════════════════════════════════════════════════


def test_canopy_water_snow_mass_balance():
    """Mass-balance error must be near zero for all canopy_water_snow fixtures."""
    for c in WATER["canopy_water_snow"]:
        inp = c["inputs"]
        state = CanopyWaterState(
            CanopyStorage=jnp.array(inp["CanopyStorage_in"]),
            SWE=jnp.array(inp["SWE_in"]),
            swe_i=jnp.array(inp["swe_i_in"]),
            swe_l=jnp.array(inp["swe_l_in"]),
        )
        params = CanopySnowParams(
            wmax=jnp.array(inp["wmax"]),
            wmaxsnow=jnp.array(inp["wmaxsnow"]),
            kmelt=jnp.array(inp["kmelt"]),
            kfreeze=jnp.array(inp["kfreeze"]),
            frac_snowliq=jnp.array(inp["frac_snowliq"]),
            gsoil=jnp.array(0.0),
        )
        _, flux = canopy_water_snow(
            state, params,
            tc=jnp.array(inp["T"]), pre=jnp.array(inp["Pre"]),
            ae=jnp.array(inp["AE"]), d=jnp.array(inp["D"]),
            ra=jnp.array(inp["Ra"]), u=jnp.array(inp["U"]),
            lai=jnp.array(inp["LAI"]), patm=jnp.array(inp["P"]),
            time_step=jnp.array(inp["time_step"]),
        )
        assert abs(float(flux.mbe)) < 1e-10, f"mbe={float(flux.mbe)} for T={inp['T']}"


def test_soil_water_mass_balance():
    """Mass-balance error must be near zero for all soil_water fixtures."""
    for c in WATER["soil_water"]:
        inp = c["inputs"]
        state = SoilWaterState(
            WatSto=jnp.array(inp["WatSto_in"]),
            PondSto=jnp.array(inp["PondSto_in"]),
            MaxWatSto=jnp.array(inp["MaxWatSto"]),
            MaxPondSto=jnp.array(inp["MaxPondSto"]),
            FcSto=jnp.array(inp["FcSto"]),
            Wliq=jnp.array(inp["Wliq_in"]),
            Psi=jnp.array(0.0), Sat=jnp.array(0.0),
            Kh=jnp.array(inp["Kh_in"]), beta=jnp.array(0.0),
        )
        soil_params = SoilHydroParams(
            watsat=jnp.array(inp["watsat"]), watres=jnp.array(inp["watres"]),
            alpha_van=jnp.array(inp["alpha_van"]), n_van=jnp.array(inp["n_van"]),
            ksat=jnp.array(inp["ksat"]),
        )
        _, flux, _, _, _ = soil_water(
            state, soil_params, max_poros=jnp.array(inp["max_poros"]),
            potinf=jnp.array(inp["potinf"]), tr=jnp.array(inp["tr"]),
            evap=jnp.array(inp["evap"]), latflow=jnp.array(inp["latflow"]),
            time_step=jnp.array(inp["time_step"]),
        )
        assert jnp.allclose(flux.mbe, c["output"]["mbe"], rtol=RTOL, atol=1e-10), \
            f"mbe={float(flux.mbe)} expected={c['output']['mbe']}"


def test_ground_evaporation_non_negative():
    """Ground evaporation must be non-negative."""
    result = ground_evaporation(
        tc=jnp.array(15.0), ae=jnp.array(200.0), vpd=jnp.array(800.0),
        ras=jnp.array(100.0), patm=jnp.array(101325.0),
        swe=jnp.array(0.0), beta=jnp.array(0.8),
        wat_sto=jnp.array(100.0), gsoil=jnp.array(0.01),
        time_step=jnp.array(1.0),
    )
    assert float(result) >= 0.0


def test_ground_evaporation_zero_with_snow():
    """Ground evaporation must be zero when snow is present."""
    result = ground_evaporation(
        tc=jnp.array(15.0), ae=jnp.array(200.0), vpd=jnp.array(800.0),
        ras=jnp.array(100.0), patm=jnp.array(101325.0),
        swe=jnp.array(5.0), beta=jnp.array(0.8),
        wat_sto=jnp.array(100.0), gsoil=jnp.array(0.01),
        time_step=jnp.array(1.0),
    )
    assert float(result) == 0.0


def test_canopy_water_snow_differentiable():
    """canopy_water_snow must produce finite gradients w.r.t. temperature."""
    state = CanopyWaterState(
        CanopyStorage=jnp.array(0.5), SWE=jnp.array(0.0),
        swe_i=jnp.array(0.0), swe_l=jnp.array(0.0),
    )
    params = CanopySnowParams(
        wmax=jnp.array(0.5), wmaxsnow=jnp.array(4.0),
        kmelt=jnp.array(2.31e-8), kfreeze=jnp.array(5.79e-9),
        frac_snowliq=jnp.array(0.05), gsoil=jnp.array(0.01),
    )
    def loss(tc):
        new_state, _ = canopy_water_snow(
            state, params, tc=tc, pre=jnp.array(1.4e-4),
            ae=jnp.array(160.0), d=jnp.array(800.0),
            ra=jnp.array(33.0), u=jnp.array(2.0),
            lai=jnp.array(3.0), patm=jnp.array(101325.0),
            time_step=jnp.array(1.0),
        )
        return new_state.CanopyStorage + new_state.SWE

    g = jax.grad(loss)(jnp.array(10.0))
    assert jnp.isfinite(g)


def test_soil_water_differentiable():
    """soil_water must produce finite gradients w.r.t. potential infiltration."""
    state = SoilWaterState(
        WatSto=jnp.array(300.0), PondSto=jnp.array(0.0),
        MaxWatSto=jnp.array(750.0), MaxPondSto=jnp.array(10.0),
        FcSto=jnp.array(200.0), Wliq=jnp.array(0.3),
        Psi=jnp.array(0.0), Sat=jnp.array(0.0),
        Kh=jnp.array(1e-6), beta=jnp.array(0.0),
    )
    soil_params = SoilHydroParams(
        watsat=jnp.array(0.75), watres=jnp.array(0.0),
        alpha_van=jnp.array(4.45), n_van=jnp.array(1.12),
        ksat=jnp.array(1e-5),
    )
    def loss(potinf):
        new_state, _, _, _, _ = soil_water(
            state, soil_params, max_poros=jnp.array(0.75),
            potinf=potinf, tr=jnp.array(0.5), evap=jnp.array(0.3),
            latflow=jnp.array(0.0), time_step=jnp.array(1.0),
        )
        return new_state.WatSto

    g = jax.grad(loss)(jnp.array(5.0))
    assert jnp.isfinite(g)


def test_canopy_precip_monotonicity():
    """More precipitation should increase throughfall."""
    state = CanopyWaterState(
        CanopyStorage=jnp.array(0.0), SWE=jnp.array(0.0),
        swe_i=jnp.array(0.0), swe_l=jnp.array(0.0),
    )
    params = CanopySnowParams(
        wmax=jnp.array(0.5), wmaxsnow=jnp.array(4.0),
        kmelt=jnp.array(2.31e-8), kfreeze=jnp.array(5.79e-9),
        frac_snowliq=jnp.array(0.05), gsoil=jnp.array(0.01),
    )
    precs = jnp.array([0.5e-4, 1.0e-4, 2.0e-4, 5.0e-4])
    trfalls = []
    for pre in precs:
        _, flux = canopy_water_snow(
            state, params, tc=jnp.array(15.0), pre=pre,
            ae=jnp.array(160.0), d=jnp.array(800.0),
            ra=jnp.array(33.0), u=jnp.array(2.0),
            lai=jnp.array(3.0), patm=jnp.array(101325.0),
            time_step=jnp.array(1.0),
        )
        trfalls.append(float(flux.Throughfall))
    diffs = [trfalls[i+1] - trfalls[i] for i in range(len(trfalls)-1)]
    assert all(d > 0 for d in diffs), f"throughfall should increase with precip: {trfalls}"


def test_canopy_water_snow_cold_dry_repeat_stays_finite():
    """Repeated cold dry steps must not drive canopy state to NaN.

    This specifically guards the JAX-only hazard where canopy evaporation can
    leave storage at ``-eps`` and the next call evaluates ``0**(-0.4)`` in the
    exposure coefficient before a branch masks it.
    """
    state = CanopyWaterState(
        CanopyStorage=jnp.array(0.0), SWE=jnp.array(0.0),
        swe_i=jnp.array(0.0), swe_l=jnp.array(0.0),
    )
    params = CanopySnowParams(
        wmax=jnp.array(0.5), wmaxsnow=jnp.array(4.0),
        kmelt=jnp.array(2.31e-8), kfreeze=jnp.array(5.79e-9),
        frac_snowliq=jnp.array(0.05), gsoil=jnp.array(0.01),
    )

    for _ in range(8):
        state, flux = canopy_water_snow(
            state, params,
            tc=jnp.array(-5.0),
            pre=jnp.array(0.0),
            ae=jnp.array(50.0),
            d=jnp.array(400.0),
            ra=jnp.array(33.0),
            u=jnp.array(2.0),
            lai=jnp.array(0.7),
            patm=jnp.array(101325.0),
            time_step=jnp.array(1.0),
        )
        assert jnp.isfinite(state.CanopyStorage)
        assert jnp.isfinite(state.SWE)
        assert jnp.isfinite(flux.CanopyEvap)
        assert float(state.CanopyStorage) >= 0.0


def test_canopy_water_snow_tiny_storage_cold_dry_stays_finite():
    """Tiny positive storage on the sublimation path must remain finite."""
    state = CanopyWaterState(
        CanopyStorage=jnp.array(1.0e-20), SWE=jnp.array(0.0),
        swe_i=jnp.array(0.0), swe_l=jnp.array(0.0),
    )
    params = CanopySnowParams(
        wmax=jnp.array(0.5), wmaxsnow=jnp.array(4.0),
        kmelt=jnp.array(2.31e-8), kfreeze=jnp.array(5.79e-9),
        frac_snowliq=jnp.array(0.05), gsoil=jnp.array(0.01),
    )

    new_state, flux = canopy_water_snow(
        state, params,
        tc=jnp.array(-5.0),
        pre=jnp.array(0.0),
        ae=jnp.array(50.0),
        d=jnp.array(400.0),
        ra=jnp.array(33.0),
        u=jnp.array(2.0),
        lai=jnp.array(0.7),
        patm=jnp.array(101325.0),
        time_step=jnp.array(1.0),
    )

    assert jnp.isfinite(new_state.CanopyStorage)
    assert jnp.isfinite(flux.CanopyEvap)
    assert float(new_state.CanopyStorage) >= 0.0
