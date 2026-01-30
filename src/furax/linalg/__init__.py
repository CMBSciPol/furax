"""Linear algebra utilities for furax operators."""

from ._lobpcg import LOBPCGResult, lobpcg_standard
from ._utils import (
    block_normal_like,
    block_norms,
    block_zeros_like,
    stack_pytrees,
    unstack_pytree,
)

__all__ = [
    'lobpcg_standard',
    'LOBPCGResult',
    # Block PyTree utilities
    'stack_pytrees',
    'unstack_pytree',
    'block_zeros_like',
    'block_normal_like',
    'block_norms',
]
