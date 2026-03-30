"""Shared helpers for Qvidja reference replay inputs."""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

_YASSO_PARAM = (
    0.51, 5.19, 0.13, 0.1, 0.5, 0.0, 1.0, 1.0, 0.99, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.163, 0.0, -0.0, 0.0, 0.0, 0.0,
    0.0, 0.158, -0.002, 0.17, -0.005, 0.067, -0.0, -1.44,
    -2.0, -6.9, 0.0042, 0.0015, -2.55, 1.24, 0.25,
)


def build_qvidja_run_kwargs(ref: Mapping[str, Any], ndays: int) -> dict[str, Any]:
    """Build the shared run_integration kwargs for the Qvidja replay."""
    defaults = ref["defaults"]
    hourly = ref["hourly"]
    daily = ref["daily"]
    nhours = ndays * 24

    return {
        "hourly_temp": jnp.array(hourly["temp_hr"][:nhours]).reshape(ndays, 24),
        "hourly_rg": jnp.array(hourly["rg_hr"][:nhours]).reshape(ndays, 24),
        "hourly_prec": jnp.array(hourly["prec_hr"][:nhours]).reshape(ndays, 24),
        "hourly_vpd": jnp.array(hourly["vpd_hr"][:nhours]).reshape(ndays, 24),
        "hourly_pres": jnp.array(hourly["pres_hr"][:nhours]).reshape(ndays, 24),
        "hourly_co2": jnp.array(hourly["co2_hr"][:nhours]).reshape(ndays, 24),
        "hourly_wind": jnp.array(hourly["wind_hr"][:nhours]).reshape(ndays, 24),
        "daily_lai": jnp.array(daily["lai_day"][:ndays]),
        "daily_manage_type": jnp.array([float(x) for x in daily["manage_type"][:ndays]]),
        "daily_manage_c_in": jnp.array(daily["manage_c_in"][:ndays]),
        "daily_manage_c_out": jnp.array(daily["manage_c_out"][:ndays]),
        "conductivity": defaults["conductivity"],
        "psi50": defaults["psi50"],
        "b_param": defaults["b"],
        "alpha_cost": defaults["alpha"],
        "gamma_cost": defaults["gamma"],
        "rdark": defaults["rdark"],
        "soil_depth": defaults["soil_depth"],
        "max_poros": defaults["max_poros"],
        "fc": defaults["fc"],
        "wp": defaults["wp"],
        "ksat": defaults["ksat"],
        "n_van": 1.14,
        "watres": 0.0,
        "alpha_van": 5.92,
        "watsat": defaults["max_poros"],
        "maxpond": 0.0,
        "wmax": 0.5,
        "wmaxsnow": 4.5,
        "kmelt": 2.8934e-5,
        "kfreeze": 5.79e-6,
        "frac_snowliq": 0.05,
        "gsoil": 5.0e-3,
        "hc": 0.6,
        "w_leaf": 0.01,
        "rw": 0.20,
        "rwmin": 0.02,
        "zmeas": 2.0,
        "zground": 0.1,
        "zo_ground": 0.01,
        "cratio_resp": defaults["cratio_resp"],
        "cratio_leaf": defaults["cratio_leaf"],
        "cratio_root": defaults["cratio_root"],
        "cratio_biomass": defaults["cratio_biomass"],
        "harvest_index": defaults["harvest_index"],
        "turnover_cleaf": defaults["turnover_cleaf"],
        "turnover_croot": defaults["turnover_croot"],
        "sla": defaults["sla"],
        "q10": defaults["q10"],
        "invert_option": defaults["invert_option"],
        "pft_is_oat": 0.0,
        "yasso_param": jnp.array(_YASSO_PARAM),
        "yasso_totc": defaults["yasso_totc"],
        "yasso_cn_input": defaults["yasso_cn_input"],
        "yasso_fract_root": defaults["yasso_fract_root"],
        "yasso_fract_legacy": 0.0,
        "yasso_tempr_c": 5.4,
        "yasso_precip_day": 1.87,
        "yasso_tempr_ampl": 20.0,
    }