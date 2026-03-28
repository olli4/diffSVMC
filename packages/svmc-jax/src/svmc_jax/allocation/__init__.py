"""Carbon allocation module — JAX implementation.

Fortran reference: vendor/SVMC/src/allocation.f90
"""

from svmc_jax.allocation.alloc_hypothesis_2 import alloc_hypothesis_2
from svmc_jax.allocation.invert_alloc import invert_alloc

__all__ = ["alloc_hypothesis_2", "invert_alloc"]
