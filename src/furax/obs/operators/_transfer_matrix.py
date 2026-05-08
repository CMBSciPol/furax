"""Pure-JAX transfer matrix method for multilayer optical stacks.

Computes Mueller matrices for multilayer birefringent stacks such as
sapphire HWP stacks.  ``frequency`` and ``incidence_angle`` are JAX-traced
so the computation is JIT-compatible and supports ``vmap`` over either
quantity.  All stack parameters are static Python dataclass fields.

Reference:
    Jones (1941), "A new calculus for the treatment of optical systems";
    Goldstein (2003), "Polarized Light".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Complex, Float

_C = 2.998e8  # speed of light, m/s

# Pauli-like basis for the Jones → Mueller formula:
#   M[i,j] = Re( Tr(Σᵢ J Σⱼ J†) ) / 2
# Replaces the original double Python for-loop with a single vectorised einsum.
_SIGMA = jnp.array(
    [
        [[1, 0], [0, 1]],  # I
        [[1, 0], [0, -1]],  # Q
        [[0, 1], [1, 0]],  # U
        [[0, -1j], [1j, 0]],  # V
    ],
    dtype=jnp.complex128,
)

__all__ = [
    'Material',
    'Stack',
    'SO_MF_HWP_STACK',
    'SO_HF_HWP_STACK',
    'mueller_matrix',
]


# ---------------------------------------------------------------------------
# Physical types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Material:
    """Optical properties of an isotropic or uniaxial material.

    Attributes:
        n_o: Ordinary refractive index.
        n_e: Extraordinary refractive index (equal to ``n_o`` for isotropic).
        loss_o: Ordinary loss tangent Im(ε)/Re(ε).
        loss_e: Extraordinary loss tangent.
    """

    n_o: float
    n_e: float
    loss_o: float = 0.0
    loss_e: float = 0.0

    @classmethod
    def isotropic(cls, n: float, loss: float = 0.0) -> Material:
        return cls(n_o=n, n_e=n, loss_o=loss, loss_e=loss)


@dataclass(frozen=True)
class Stack:
    """Multilayer optical stack.

    Attributes:
        thicknesses: Layer thicknesses in metres.
        materials: Material for each layer.
        crystal_angles: Extraordinary-axis orientations (radians from the
            x-axis) for uniaxial layers; zero for isotropic layers.
    """

    thicknesses: tuple[float, ...]
    materials: tuple[Material, ...]
    crystal_angles: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.thicknesses)
        if len(self.materials) != n or len(self.crystal_angles) != n:
            raise ValueError('thicknesses, materials and crystal_angles must have equal length')

    @classmethod
    def build(
        cls,
        thicknesses: Sequence[float],
        materials: Sequence[Material],
        crystal_angles: Sequence[float],
    ) -> Stack:
        return cls(
            thicknesses=tuple(thicknesses),
            materials=tuple(materials),
            crystal_angles=tuple(crystal_angles),
        )


# ---------------------------------------------------------------------------
# Simons Observatory HWP stacks
# ---------------------------------------------------------------------------

_MM = 1e-3  # metres per millimetre
_DEG = np.pi / 180  # radians per degree

# Materials (shared between MF and HF)
_SAPPHIRE = Material(n_o=3.05, n_e=3.38, loss_o=2.3e-4, loss_e=1.25e-4)
_DUROID = Material.isotropic(n=1.41, loss=1.2e-3)
_MULLITE = Material.isotropic(n=2.52, loss=0.0121)
_EPOTECK = Material.isotropic(n=1.70, loss=0.0)

_SO_MATERIALS = (
    _DUROID,
    _EPOTECK,
    _MULLITE,
    _SAPPHIRE,
    _SAPPHIRE,
    _SAPPHIRE,
    _MULLITE,
    _EPOTECK,
    _DUROID,
)

SO_MF_HWP_STACK = Stack(
    thicknesses=(
        0.394 * _MM,
        0.04 * _MM,
        0.212 * _MM,
        3.75 * _MM,
        3.75 * _MM,
        3.75 * _MM,
        0.212 * _MM,
        0.04 * _MM,
        0.394 * _MM,
    ),
    materials=_SO_MATERIALS,
    crystal_angles=(0.0, 0.0, 0.0, 0.0, 54.0 * _DEG, 0.0, 0.0, 0.0, 0.0),
)
"""Simons Observatory MF (85–145 GHz) HWP stack."""

SO_HF_HWP_STACK = Stack(
    thicknesses=(
        0.183 * _MM,
        0.04 * _MM,
        0.097 * _MM,
        1.60 * _MM,
        1.60 * _MM,
        1.60 * _MM,
        0.097 * _MM,
        0.04 * _MM,
        0.183 * _MM,
    ),
    materials=_SO_MATERIALS,
    crystal_angles=(0.0, 0.0, 0.0, 0.0, 57.0 * _DEG, 0.0, 0.0, 0.0, 0.0),
)
"""Simons Observatory HF (225–280 GHz) HWP stack."""


# ---------------------------------------------------------------------------
# Core computation: pure JAX functions
# ---------------------------------------------------------------------------


def _jones_to_mueller(J: Array) -> Float[Array, '4 4']:
    """Convert a 2×2 Jones matrix to a 4×4 real Mueller matrix.

    Vectorised replacement for the original double Python for-loop.
    """
    # M[i,j] = Re( Tr(Σᵢ J Σⱼ J†) ) / 2
    # = Re( Σᵢ[a,b] J[b,c] Σⱼ[c,d] conj(J)[a,d] ) / 2
    return jnp.einsum('iab,bc,jcd,ad->ij', _SIGMA, J, _SIGMA, jnp.conj(J)).real / 2


def _layer_transfer_matrix(
    material: Material,
    thickness: float,
    frequency: Array,
    nsin: Array,
    chi: float,
) -> Complex[Array, '4 4']:
    """4×4 complex transfer matrix for a single anisotropic layer.

    Args:
        material: Optical properties of the layer.
        thickness: Layer thickness in metres (static).
        frequency: Frequency in Hz (JAX-traced).
        nsin: Snell's-law invariant n₀ sin θ₀ (JAX-traced).
        chi: Extraordinary-axis orientation in radians (static).
    """
    nO = float(material.n_o)
    nEmat = float(material.n_e)

    # Effective extraordinary index corrected for oblique incidence
    nE = nEmat * jnp.sqrt(1 + (nEmat**-2 - nO**-2) * nsin**2 * np.cos(chi) ** 2)

    thetaO = jnp.arcsin(nsin / nO)
    thetaE = jnp.arcsin(nsin / nE)

    # Complex refractive indices (loss included)
    n_o_c = jnp.sqrt(jnp.array((1 - 1j * material.loss_o) * nO**2, dtype=jnp.complex128))
    n_e_c = jnp.sqrt(jnp.array(1 - 1j * material.loss_e, dtype=jnp.complex128) * nE**2)

    k0 = 2 * jnp.pi * frequency / _C

    # ------------------------------------------------------------------
    # Rotated dielectric tensor (3×3 complex)
    # rot is constant at JIT time since chi is static
    # ------------------------------------------------------------------
    c_chi, s_chi = float(np.cos(chi)), float(np.sin(chi))
    rot = jnp.array(
        [[c_chi, -s_chi, 0.0], [s_chi, c_chi, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.complex128,
    )
    eps = jnp.diag(jnp.array([nE**2, nO**2, nO**2], dtype=jnp.complex128))
    eps_rot = rot @ eps @ rot.conj().T  # rot.T = rot.conj().T for real chi
    eps_rot_inv = jnp.linalg.inv(eps_rot)

    # ------------------------------------------------------------------
    # Field polarisation vectors for ordinary and extraordinary rays
    # ------------------------------------------------------------------
    # Ordinary displacement D
    denom_Do = jnp.sqrt(jnp.cos(thetaO) ** 2 + jnp.sin(thetaO) ** 2 * s_chi**2)
    D_o = (
        jnp.array(
            [
                -s_chi * jnp.cos(thetaO),
                c_chi * jnp.cos(thetaO),
                s_chi * jnp.sin(thetaO),
            ],
            dtype=jnp.complex128,
        )
        / denom_Do
    )

    # Ordinary magnetic H
    denom_Ho = jnp.sqrt(jnp.cos(thetaO) ** 2 * c_chi**2 + s_chi**2)
    H_o = (
        jnp.array(
            [
                -(jnp.cos(thetaO) ** 2) * c_chi,
                -s_chi,
                jnp.cos(thetaO) * jnp.sin(thetaO) * c_chi,
            ],
            dtype=jnp.complex128,
        )
        / denom_Ho
    )

    # Extraordinary displacement D
    denom_De = jnp.sqrt(c_chi**2 * jnp.cos(thetaO) ** 2 + s_chi**2 * jnp.cos(thetaO - thetaE) ** 2)
    D_e = (
        jnp.array(
            [
                c_chi * jnp.cos(thetaO) * jnp.cos(thetaE),
                s_chi * (jnp.sin(thetaO) * jnp.sin(thetaE) + jnp.cos(thetaO) * jnp.cos(thetaE)),
                -c_chi * jnp.cos(thetaE) * jnp.sin(thetaO),
            ],
            dtype=jnp.complex128,
        )
        / denom_De
    )

    # Extraordinary magnetic H
    denom_He = jnp.sqrt(jnp.cos(thetaO - thetaE) ** 2 * s_chi**2 + jnp.cos(thetaO) ** 2 * c_chi**2)
    H_e = (
        jnp.array(
            [
                -jnp.cos(thetaO - thetaE) * jnp.cos(thetaE) * s_chi,
                jnp.cos(thetaO) * c_chi,
                jnp.cos(thetaO - thetaE) * jnp.sin(thetaE) * s_chi,
            ],
            dtype=jnp.complex128,
        )
        / denom_He
    )

    # ------------------------------------------------------------------
    # Phi: relates boundary tangential fields to wave amplitudes
    # ------------------------------------------------------------------
    Phi = jnp.array(
        [
            [D_o[0], D_e[0], D_o[0], D_e[0]],
            [H_o[1] / nO, H_e[1] / nE, -H_o[1] / nO, -H_e[1] / nE],
            [D_o[1], D_e[1], D_o[1], D_e[1]],
            [-H_o[0] / nO, -H_e[0] / nE, H_o[0] / nO, H_e[0] / nE],
        ],
        dtype=jnp.complex128,
    )

    # Phase accumulated crossing the layer
    delta_o = 1j * k0 * n_o_c * thickness * jnp.cos(thetaO)
    delta_e = 1j * k0 * n_e_c * thickness * jnp.cos(thetaE)
    P = jnp.diag(
        jnp.array(
            [jnp.exp(-delta_o), jnp.exp(-delta_e), jnp.exp(delta_o), jnp.exp(delta_e)],
            dtype=jnp.complex128,
        )
    )

    # Psi: D-field to E-field conversion via inverse dielectric tensor
    Psi = jnp.array(
        [
            [eps_rot_inv[0, 0], 0, eps_rot_inv[0, 1], 0],
            [0, 1, 0, 0],
            [eps_rot_inv[1, 0], 0, eps_rot_inv[1, 1], 0],
            [0, 0, 0, 1],
        ],
        dtype=jnp.complex128,
    )

    # Transfer matrix: Ψ Φ (Ψ Φ P)⁻¹
    PsiPhi = Psi @ Phi
    return PsiPhi @ jnp.linalg.inv(PsiPhi @ P)  # type: ignore[no-any-return]


def _stack_transfer_matrix(
    stack: Stack,
    frequency: Array,
    incidence_angle: Array,
    rotation: float = 0.0,
    input_index: float = 1.0,
) -> Complex[Array, '4 4']:
    """Product of all layer transfer matrices (left to right)."""
    nsin = jnp.sin(incidence_angle) * input_index
    total = jnp.eye(4, dtype=jnp.complex128)
    for mat, thick, crystal_angle in zip(stack.materials, stack.thicknesses, stack.crystal_angles):
        total = total @ _layer_transfer_matrix(
            mat, thick, frequency, nsin, crystal_angle + rotation
        )
    return total


def _transfer_to_jones(
    total: Array,
    incidence_angle: Array,
    input_index: float,
    exit_index: float,
) -> tuple[Complex[Array, '2 2'], Complex[Array, '2 2']]:
    """Transmitted and reflected Jones matrices from the total transfer matrix."""
    nsin = jnp.sin(incidence_angle) * input_index
    exit_angle = jnp.arcsin(nsin / exit_index)
    m = total
    n1, n3 = float(input_index), float(exit_index)
    th1, th3 = incidence_angle, exit_angle

    A = (m[0, 0] * jnp.cos(th3) + m[0, 1] * n3) / jnp.cos(th1)
    B = (m[0, 2] + m[0, 3] * n3 * jnp.cos(th3)) / jnp.cos(th1)
    C = (m[1, 0] * jnp.cos(th3) + m[1, 1] * n3) / n1
    D = (m[1, 2] + m[1, 3] * n3 * jnp.cos(th3)) / n1
    N = m[2, 0] * jnp.cos(th3) + m[2, 1] * n3
    K = m[2, 2] + m[2, 3] * n3 * jnp.cos(th3)
    Pv = (m[3, 0] * jnp.cos(th3) + m[3, 1] * n3) / (n1 * jnp.cos(th1))
    S = (m[3, 2] + m[3, 3] * n3 * jnp.cos(th3)) / (n1 * jnp.cos(th1))

    denom = (A + C) * (K + S) - (B + D) * (N + Pv)
    J_tran = jnp.array([[K + S, -B - D], [-N - Pv, A + C]], dtype=jnp.complex128) * 2 / denom
    J_refl = (
        jnp.array(
            [
                [(C - A) * (K + S) - (D - B) * (N + Pv), 2 * (A * D - C * B)],
                [2 * (N * S - Pv * K), (A + C) * (K - S) - (D + B) * (N - Pv)],
            ],
            dtype=jnp.complex128,
        )
        / denom
    )
    return J_tran, J_refl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mueller_matrix(
    stack: Stack,
    frequency: Array,
    incidence_angle: Array,
    rotation: float = 0.0,
    input_index: float = 1.0,
    exit_index: float = 1.0,
    *,
    reflected: bool = False,
) -> Float[Array, '4 4']:
    """4×4 Mueller matrix for light through a multilayer optical stack.

    JIT-friendly: ``frequency`` and ``incidence_angle`` are JAX-traced.
    Use ``jax.vmap`` to batch over either quantity efficiently.

    Args:
        stack: Multilayer stack (static Python dataclass).
        frequency: Frequency in Hz.
        incidence_angle: Angle of incidence in radians.
        rotation: Stack rotation about the optical axis in radians.
        input_index: Refractive index of the input medium.
        exit_index: Refractive index of the exit medium.
        reflected: Return the reflected Mueller matrix instead of transmitted.

    Returns:
        4×4 real Mueller matrix.

    Example::

        import jax
        import jax.numpy as jnp
        from furax.obs.operators import mueller_matrix, Material, Stack

        sapphire = Material(n_o=3.05, n_e=3.38, loss_o=2.3e-4, loss_e=1.25e-4)
        stack = Stack.build(
            thicknesses=[3.75e-3, 3.75e-3, 3.75e-3],
            materials=[sapphire, sapphire, sapphire],
            crystal_angles=[0.0, jnp.deg2rad(54.0), 0.0],
        )

        M = mueller_matrix(stack, frequency=150e9, incidence_angle=jnp.deg2rad(5.0))

        # Batched over detectors with different incident angles:
        angles = jnp.deg2rad(jnp.linspace(0, 10, 64))
        Ms = jax.vmap(lambda a: mueller_matrix(stack, 150e9, a))(angles)
    """
    total = _stack_transfer_matrix(stack, frequency, incidence_angle, rotation, input_index)
    J_tran, J_refl = _transfer_to_jones(total, incidence_angle, input_index, exit_index)
    return _jones_to_mueller(J_refl if reflected else J_tran)
