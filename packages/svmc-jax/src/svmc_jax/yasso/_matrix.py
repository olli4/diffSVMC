"""Shared AWENH matrix assembly helpers for Yasso modules."""

import jax.numpy as jnp


_TRANSFER_ROWS = jnp.array([
    0, 0, 0,
    1, 1, 1,
    2, 2, 2,
    3, 3, 3,
    4, 4, 4, 4,
], dtype=jnp.int32)
_TRANSFER_COLS = jnp.array([
    1, 2, 3,
    0, 2, 3,
    0, 1, 3,
    0, 1, 2,
    0, 1, 2, 3,
], dtype=jnp.int32)
_SOURCE_POOLS = jnp.array([
    1, 2, 3,
    0, 2, 3,
    0, 1, 3,
    0, 1, 2,
    0, 1, 2, 3,
], dtype=jnp.int32)
_LEACH_DIAG = jnp.array([0, 1, 2, 3], dtype=jnp.int32)


def build_awenh_matrix(
    diag_rates: jnp.ndarray,
    transfer_params: jnp.ndarray,
    humus_transfer: jnp.ndarray,
    *,
    leac_term: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Assemble the 5×5 AWENH decomposition matrix from diagonal rates.

    Args:
        diag_rates: Poolwise diagonal decomposition rates with shape (5,).
        transfer_params: The 12 AWEN transfer coefficients ``param[4:16]``.
        humus_transfer: Shared transfer coefficient from AWEN pools to H.
        leac_term: Optional diagonal addition for AWEN pools only.

    Returns:
        5×5 decomposition matrix.
    """
    abs_diag = jnp.abs(diag_rates)
    awen_values = transfer_params * abs_diag[_SOURCE_POOLS[:12]]
    humus_values = humus_transfer * abs_diag[_SOURCE_POOLS[12:]]
    transfer_values = jnp.concatenate([awen_values, humus_values])

    matrix = jnp.diag(diag_rates)
    matrix = matrix.at[_TRANSFER_ROWS, _TRANSFER_COLS].set(transfer_values)

    if leac_term is not None:
        matrix = matrix.at[_LEACH_DIAG, _LEACH_DIAG].add(leac_term)

    return matrix