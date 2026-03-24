"""Yasso20 soil carbon decomposition — leaf-level functions in JAX."""

import json
from pathlib import Path

import jax.numpy as jnp

_CONSTANTS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "svmc-ref" / "constants"
_YASSO_CONSTANTS = json.loads((_CONSTANTS_DIR / "yasso.json").read_text())

# AWENH fractions — canonical source: packages/svmc-ref/constants/yasso.json
AWENH_LEAF     = jnp.array(_YASSO_CONSTANTS["AWENH_LEAF"])
AWENH_FINEROOT = jnp.array(_YASSO_CONSTANTS["AWENH_FINEROOT"])
AWENH_SOLUBLE  = jnp.array(_YASSO_CONSTANTS["AWENH_SOLUBLE"])
AWENH_COMPOST  = jnp.array(_YASSO_CONSTANTS["AWENH_COMPOST"])


def inputs_to_fractions(
    leaf: float,
    root: float,
    soluble: float,
    compost: float,
) -> jnp.ndarray:
    """Split carbon inputs into AWENH pool fractions.

    Fortran: inputs_to_fractions in yasso.f90

    The fifth pool (H — Humus) never receives external input.

    Args:
        leaf: Leaf litter carbon input.
        root: Fine-root litter carbon input.
        soluble: Soluble carbon input.
        compost: Compost carbon input.

    Returns:
        Array of length 5 — AWENH fractions.
    """
    return (leaf * AWENH_LEAF
            + root * AWENH_FINEROOT
            + soluble * AWENH_SOLUBLE
            + compost * AWENH_COMPOST)
