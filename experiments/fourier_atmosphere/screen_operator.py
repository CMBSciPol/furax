"""Helpers shared by the experiment scripts, on top of `furax.obs.atmosphere`."""

import jax.numpy as jnp
import numpy as np

from furax.obs.atmosphere import FourierScreenOperator, cell_index, fit_cell
from furax.obs.coords import ZAXIS

__all__ = [
    'FourierScreenOperator',
    'cell_index',
    'fit_cell',
    'half_plane_index',
    'screen_coords',
    'von_karman',
]


def half_plane_index(box: float, q_max: float) -> np.ndarray:
    """Integer wavevectors n of a square box, |n| <= int(q_max box / 2 pi), one per +-n pair."""
    n = int(q_max * box / (2 * np.pi))
    nx, ny = np.meshgrid(np.arange(-n, n + 1), np.arange(0, n + 1))
    nx, ny = nx.ravel(), ny.ravel()
    keep = ((ny > 0) | (nx > 0)) & (nx**2 + ny**2 <= n**2)
    return np.stack([nx[keep], ny[keep]], -1)


def von_karman(modes, amplitude: float, q0: float):
    return amplitude * (jnp.sum(modes**2, -1) + q0**2) ** (-11 / 6)


def screen_coords(qbore, qdet):
    """Screen coordinates (before drift) of every sample, shape (det, samp, 2)."""
    v = (qbore[None, :] * qdet[:, None]).rotate_vector(ZAXIS)
    return v[..., :2] / v[..., 2:]
