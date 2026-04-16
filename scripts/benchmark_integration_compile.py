"""Lightweight compile/runtime benchmark for grouped JAX integration.

Measures:
1. JIT compile + first execution for a single grouped integration run
2. Steady-state runtime for the same compiled single-site run
3. JIT compile + first execution for a vmapped grouped batch run
4. Steady-state runtime for the compiled vmapped run

Defaults are intentionally small so the script is practical during normal
development. Use ``--platform default`` to benchmark the current default
device instead of the CPU-focused baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
import sys


def _preparse_platform(argv: list[str]) -> str:
    for index, token in enumerate(argv):
        if token == "--platform" and index + 1 < len(argv):
            return argv[index + 1]
    return "cpu"


if _preparse_platform(sys.argv[1:]) == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SVMC_JAX_SRC = REPO_ROOT / "packages" / "svmc-jax" / "src"
if str(SVMC_JAX_SRC) not in sys.path:
    sys.path.insert(0, str(SVMC_JAX_SRC))

from svmc_jax.integration import run_integration_grouped
from svmc_jax.qvidja_replay import build_qvidja_run_inputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark grouped JAX integration compile and run time.",
    )
    parser.add_argument("--days", type=int, default=7, help="Replay length in days.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for vmapped run.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup repeats before timing steady-state.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed repeats for steady-state runs.")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only measure true compilation via lower(...).compile(); skip steady-state timing.",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "batch", "both"),
        default="both",
        help="Benchmark the single run, vmapped batch run, or both (default).",
    )
    parser.add_argument(
        "--platform",
        choices=("cpu", "default"),
        default="cpu",
        help="Benchmark on CPU for repeatability, or on the current default device.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable summary.",
    )
    return parser.parse_args()


def _block_tree(tree):
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def _measure_call(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    _block_tree(result)
    return time.perf_counter() - start, result


def _measure_compile(fn, *args):
    start = time.perf_counter()
    compiled = fn.lower(*args).compile()
    return time.perf_counter() - start, compiled


def _measure_steady_state(fn, *args, warmup: int, repeats: int):
    for _ in range(warmup):
        result = fn(*args)
        _block_tree(result)

    durations = []
    for _ in range(repeats):
        duration, result = _measure_call(fn, *args)
        durations.append(duration)

    return {
        "median_s": statistics.median(durations),
        "min_s": min(durations),
        "max_s": max(durations),
        "last_result": result,
    }


def _make_mode_flags(mode: str) -> tuple[bool, bool]:
    return mode in ("single", "both"), mode in ("batch", "both")


def _load_reference():
    ref_path = REPO_ROOT / "website" / "public" / "qvidja-v1-reference.json"
    with open(ref_path) as handle:
        return json.load(handle)


def _broadcast_leaf(x, batch: int):
    arr = jnp.asarray(x)
    return jnp.broadcast_to(arr, (batch,) + arr.shape)


def _make_batched_inputs(forcing, params, batch: int):
    batched_forcing = jax.tree_util.tree_map(lambda x: _broadcast_leaf(x, batch), forcing)
    batched_params = jax.tree_util.tree_map(lambda x: _broadcast_leaf(x, batch), params)
    alpha_values = jnp.linspace(0.05, 0.15, batch, dtype=jnp.float64)
    batched_params = batched_params._replace(
        phydro=batched_params.phydro._replace(alpha_cost=alpha_values),
    )
    return batched_forcing, batched_params


def _select_device(platform: str):
    if platform == "default":
        return jax.devices()[0]
    return jax.devices("cpu")[0]


def main() -> None:
    args = _parse_args()
    device = _select_device(args.platform)
    ref = _load_reference()
    do_single, do_batch = _make_mode_flags(args.mode)

    if not args.compile_only and args.repeats <= 0:
        raise ValueError("--repeats must be positive unless --compile-only is used")

    with jax.default_device(device):
        forcing, params = build_qvidja_run_inputs(ref, args.days)
        single_compile_s = None
        single_compile_result = None
        single_steady = None
        batch_compile_s = None
        batch_compile_result = None
        batch_steady = None

        if do_single:
            single_run = jax.jit(lambda f, p: run_integration_grouped(f, p))
            if args.compile_only:
                single_compile_s, single_compiled = _measure_compile(single_run, forcing, params)
                single_compile_result = single_compiled(forcing, params)
                _block_tree(single_compile_result)
            else:
                single_compile_s, single_compile_result = _measure_call(single_run, forcing, params)
                single_steady = _measure_steady_state(
                    single_run,
                    forcing,
                    params,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )

        if do_batch:
            batched_forcing, batched_params = _make_batched_inputs(forcing, params, args.batch)
            batched_run = jax.jit(
                jax.vmap(lambda forcing_i, params_i: run_integration_grouped(forcing_i, params_i)),
            )
            if args.compile_only:
                batch_compile_s, batch_compiled = _measure_compile(
                    batched_run,
                    batched_forcing,
                    batched_params,
                )
                batch_compile_result = batch_compiled(batched_forcing, batched_params)
                _block_tree(batch_compile_result)
            else:
                batch_compile_s, batch_compile_result = _measure_call(
                    batched_run,
                    batched_forcing,
                    batched_params,
                )
                batch_steady = _measure_steady_state(
                    batched_run,
                    batched_forcing,
                    batched_params,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )

    if do_single:
        single_out = single_compile_result[1]
        if not jnp.all(jnp.isfinite(single_out.gpp_avg)):
            raise RuntimeError("Non-finite GPP in single-run benchmark output")
    if do_batch:
        batch_out = batch_compile_result[1]
        if not jnp.all(jnp.isfinite(batch_out.gpp_avg)):
            raise RuntimeError("Non-finite GPP in batched benchmark output")

    summary = {
        "device": str(device),
        "days": args.days,
        "batch": args.batch,
        "mode": args.mode,
        "compile_only": args.compile_only,
        "single_compile_s": single_compile_s,
        "single_steady_median_s": None if single_steady is None else single_steady["median_s"],
        "single_steady_min_s": None if single_steady is None else single_steady["min_s"],
        "single_steady_max_s": None if single_steady is None else single_steady["max_s"],
        "batch_compile_s": batch_compile_s,
        "batch_steady_median_s": None if batch_steady is None else batch_steady["median_s"],
        "batch_steady_min_s": None if batch_steady is None else batch_steady["min_s"],
        "batch_steady_max_s": None if batch_steady is None else batch_steady["max_s"],
        "batch_fields_per_second": None if batch_steady is None else args.batch / batch_steady["median_s"],
        "batch_ms_per_field": None if batch_steady is None else 1000.0 * batch_steady["median_s"] / args.batch,
        "single_daily_gpp_last": None if not do_single else float(single_out.gpp_avg[-1]),
        "batch_daily_gpp_last_mean": None if not do_batch else float(jnp.mean(batch_out.gpp_avg[:, -1])),
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(f"Device: {summary['device']}")
    print(f"Replay length: {args.days} day(s)")
    print(f"Batch size: {args.batch}")
    print(f"Mode: {args.mode}")

    if do_single:
        print("\nSingle-site grouped run")
        if args.compile_only:
            print(f"  compile only: {summary['single_compile_s']:.3f} s")
        else:
            print(f"  compile + first run: {summary['single_compile_s']:.3f} s")
        if args.compile_only:
            print("  steady-state timing: skipped (--compile-only)")
        else:
            print(
                "  steady-state median: "
                f"{summary['single_steady_median_s']*1000:.1f} ms "
                f"(min {summary['single_steady_min_s']*1000:.1f}, max {summary['single_steady_max_s']*1000:.1f})"
            )

    if do_batch:
        print("\nVmapped grouped run")
        if args.compile_only:
            print(f"  compile only: {summary['batch_compile_s']:.3f} s")
        else:
            print(f"  compile + first run: {summary['batch_compile_s']:.3f} s")
        if args.compile_only:
            print("  steady-state timing: skipped (--compile-only)")
        else:
            print(
                "  steady-state median: "
                f"{summary['batch_steady_median_s']*1000:.1f} ms "
                f"(min {summary['batch_steady_min_s']*1000:.1f}, max {summary['batch_steady_max_s']*1000:.1f})"
            )
            print(f"  throughput: {summary['batch_fields_per_second']:.1f} fields/s")
            print(f"  ms/field: {summary['batch_ms_per_field']:.2f}")


if __name__ == "__main__":
    main()