#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from netCDF4 import Dataset, num2date


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "vendor" / "SVMC" / "data" / "input"
OUTPUT_PATH = ROOT / "website" / "public" / "qvidja-v1-reference.json"

START_DATE = datetime(2018, 5, 10)
END_DATE_EXCLUSIVE = datetime(2023, 1, 1)

YEARS = [2018, 2019, 2020, 2021, 2022]


@dataclass(frozen=True)
class DailySpec:
    suffix: str
    variable: str


DAILY_SPECS = {
    "lai_day": DailySpec("lai.gp", "LAI"),
    "snowdepth_day": DailySpec("snowdepth", "SnowDepth"),
    "soilmoist_day": DailySpec("soilmoist", "SoilMoist"),
}


def serialize_dates(values: list[datetime]) -> list[str]:
    return [value.strftime("%Y-%m-%dT%H:%M:%SZ") for value in values]


def load_time_axis(dataset: Dataset) -> list[datetime]:
    time_var = dataset.variables["time"]
    values = num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, "calendar", "standard"))
    return [datetime(value.year, value.month, value.day, getattr(value, "hour", 0), getattr(value, "minute", 0), getattr(value, "second", 0)) for value in values]


def flatten_first_cell(var) -> list[float]:
    data = var[:]
    while getattr(data, "ndim", 1) > 1:
      data = data[:, 0]
    return [float(value) for value in data]


def filter_series(times: list[datetime], values: list[float], *, daily: bool) -> tuple[list[datetime], list[float]]:
    filtered_times: list[datetime] = []
    filtered_values: list[float] = []
    for timestamp, value in zip(times, values, strict=True):
        if timestamp < START_DATE or timestamp >= END_DATE_EXCLUSIVE:
            continue
        if daily:
            timestamp = datetime(timestamp.year, timestamp.month, timestamp.day)
        filtered_times.append(timestamp)
        filtered_values.append(value)
    return filtered_times, filtered_values


def load_hourly_reference() -> dict[str, object]:
    fields = {
        "temp_hr": "air_temperature",
        "rg_hr": "surface_downwelling_shortwave_flux_in_air",
        "prec_hr": "precipitation_flux",
        "vpd_hr": "water_vapor_saturation_deficit",
        "pres_hr": "air_pressure",
        "co2_hr": "mole_fraction_of_carbon_dioxide_in_air",
        "wind_hr": "wind_speed",
    }

    time_axis: list[datetime] = []
    data = {name: [] for name in fields}

    for year in YEARS:
        path = INPUT_DIR / f"FieldObs_Qvidja.{year}.hr.timeshift.nc"
        with Dataset(path) as dataset:
            times = load_time_axis(dataset)
            arrays = {name: flatten_first_cell(dataset.variables[var_name]) for name, var_name in fields.items()}

        filtered_times, first_values = filter_series(times, arrays["temp_hr"], daily=False)
        time_axis.extend(filtered_times)
        data["temp_hr"].extend(first_values)

        for name, values in arrays.items():
            if name == "temp_hr":
                continue
            _, filtered_values = filter_series(times, values, daily=False)
            data[name].extend(filtered_values)

    return {
        "timestamps": serialize_dates(time_axis),
        **data,
    }


def load_daily_reference() -> dict[str, object]:
    data: dict[str, list[float] | list[int] | list[str]] = {
        "dates": [],
        "lai_day": [],
        "snowdepth_day": [],
        "soilmoist_day": [],
        "manage_type": [],
        "manage_c_in": [],
        "manage_c_out": [],
    }

    for year in YEARS:
        year_dates: list[datetime] | None = None

        for target_name, spec in DAILY_SPECS.items():
            path = INPUT_DIR / f"FieldObs_Qvidja.{year}.{spec.suffix}.nc"
            with Dataset(path) as dataset:
                times = load_time_axis(dataset)
                values = flatten_first_cell(dataset.variables[spec.variable])
            filtered_times, filtered_values = filter_series(times, values, daily=True)
            if year_dates is None:
                year_dates = filtered_times
                data["dates"].extend([dt.strftime("%Y-%m-%d") for dt in filtered_times])
            data[target_name].extend(filtered_values)

        management_path = INPUT_DIR / f"FieldObs_Qvidja.{year}.management.nc"
        with Dataset(management_path) as dataset:
            times = load_time_axis(dataset)
            manage_type = [int(round(value)) for value in flatten_first_cell(dataset.variables["management_type"])]
            manage_c_in = flatten_first_cell(dataset.variables["management_c_input"])
            manage_c_out = flatten_first_cell(dataset.variables["management_c_output"])

        _, filtered_manage_type = filter_series(times, manage_type, daily=True)
        _, filtered_manage_c_in = filter_series(times, manage_c_in, daily=True)
        _, filtered_manage_c_out = filter_series(times, manage_c_out, daily=True)

        data["manage_type"].extend(int(value) for value in filtered_manage_type)
        data["manage_c_in"].extend(filtered_manage_c_in)
        data["manage_c_out"].extend(filtered_manage_c_out)

    return data


def main() -> None:
    hourly = load_hourly_reference()
    daily = load_daily_reference()

    nhours = len(hourly["temp_hr"])
    ndays = len(daily["lai_day"])
    if nhours != ndays * 24:
        raise SystemExit(f"Expected nhours == ndays * 24, got {nhours} and {ndays}")

    payload = {
        "site": {
            "name": "Qvidja",
            "latitude": 60.2955,
            "longitude": 22.39281,
            "reference": "SVMC v1.0.0 Qvidja replay from vendored NetCDF inputs",
            "start_date": START_DATE.strftime("%Y-%m-%d"),
            "end_date_exclusive": END_DATE_EXCLUSIVE.strftime("%Y-%m-%d"),
            "nhours": nhours,
            "ndays": ndays,
        },
        "defaults": {
            "pft_type_code": 1,
            "opt_hypothesis": "PM",
            "obs_lai": True,
            "obs_soilmoist": False,
            "obs_snowdepth": False,
            "conductivity": 3e-17,
            "psi50": -4.0,
            "b": 2.0,
            "alpha": 0.08,
            "gamma": 1.0,
            "rdark": 0.015,
            "soil_depth": 0.6,
            "max_poros": 0.68,
            "fc": 0.4,
            "wp": 0.12,
            "ksat": 2e-6,
            "cratio_resp": 5e-8,
            "cratio_leaf": 0.6,
            "cratio_root": 0.4,
            "cratio_biomass": 0.42,
            "harvest_index": 0.8,
            "turnover_cleaf": 0.03,
            "turnover_croot": 0.004,
            "sla": 10.0,
            "q10": 2.0,
            "invert_option": 0,
            "yasso_totc": 16.0,
            "yasso_cn_input": 50.0,
            "yasso_fract_root": 0.6,
            "yasso_fract_legacy": 0.3,
        },
        "hourly": hourly,
        "daily": daily,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} ({nhours} hourly steps, {ndays} daily steps)")


if __name__ == "__main__":
    main()