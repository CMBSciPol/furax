from dataclasses import field
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
from fastquat import Quaternion
from jax_finufft import nufft2
from jaxtyping import Array, Bool, Float, Int

from furax import AbstractLinearOperator
from furax.obs.coords import ZAXIS
from furax.obs.pointing import QuaternionSampler, SamplingKernel

__all__ = [
    'FourierScreenOperator',
    'cell_index',
    'fit_cell',
]


class FourierScreenOperator(AbstractLinearOperator):
    r"""Detectors reading a frozen atmosphere layer represented by a band-limited Fourier series.

    The layer is a real Fourier series on a periodic cell of the screen plane, the parallelogram
    $\mathbf{y} = \mathbf{y}_0 + B\boldsymbol{\xi}$, $\boldsymbol{\xi} \in [0, 1)^2$, whose edges
    are the columns of $B$:

    $$
    m(\mathbf{y}) = \sum_{\mathbf{n}} c_{\mathbf{n}} \cos(2\pi\,\mathbf{n}\cdot\boldsymbol{\xi})
        + s_{\mathbf{n}} \sin(2\pi\,\mathbf{n}\cdot\boldsymbol{\xi}),
    $$

    with wavevectors $\mathbf{q}_{\mathbf{n}} = 2\pi B^{-\mathsf{T}}\mathbf{n}$. Screen positions
    are in units of the layer height: the line of sight of direction $(v_x, v_y, v_z)$ (horizon
    frame, $z$ = zenith) crosses the layer at $\mathbf{u} = (v_x, v_y) / v_z$, exactly for a flat
    layer. Sample $t$ reads the series at $\mathbf{u} + \boldsymbol{\omega}\,t$, the drift
    convention of [`ScreenSampler`][furax.obs.atmosphere.ScreenSampler]: the pattern moves at
    $-\boldsymbol{\omega}$, so $\boldsymbol{\omega}$ is minus the wind velocity over the
    height.

    The input is the coefficients $(c, s)$, shape (2, n_modes), and the output the samples, shape
    (n_detectors, n_samples). A sample without pointing (zero quaternion) reads zero. The
    operator is differentiable with respect to `omega` and the pointing.

    Attributes:
        index: Integer wavevector $\mathbf{n}$ of each mode, shape (n_modes, 2), one per
            $\pm\mathbf{n}$ pair, see [`cell_index`][furax.obs.atmosphere.cell_index].
        sampler: The pointing of the detectors.
        times: Time of each sample, shape (n_samples,).
        omega: Drift rate of the layer, minus the wind velocity over height, shape (2,).
        cell: The cell edges $B$ as columns, shape (2, 2).
        origin: The cell corner $\mathbf{y}_0$, shape (2,).
        n_max: Largest $|n_1|$ and $|n_2|$ of `index`.
        backend: `'nufft'` evaluates the series with a type-2 non-uniform FFT, in
            $O(N_\mathrm{samp} + M \log M)$; `'dense'` sums it directly, in
            $O(N_\mathrm{samp} M)$.
        eps: Requested relative accuracy of the NUFFT.
    """

    index: Int[Array, 'modes 2']
    sampler: QuaternionSampler
    times: Float[Array, ' samp']
    omega: Float[Array, ' 2']
    cell: Float[Array, '2 2']
    origin: Float[Array, ' 2']
    n_max: tuple[int, int] = field(metadata={'static': True})
    backend: Literal['dense', 'nufft'] = field(default='nufft', metadata={'static': True})
    eps: float = field(default=1e-9, metadata={'static': True})

    @classmethod
    def create(
        cls,
        index: Int[Array, 'modes 2'],
        cell: float | Float[Array, '2 2'],
        qbore: Quaternion,
        qdet: Quaternion,
        times: Float[Array, ' samp'],
        omega: Float[Array, ' 2'],
        backend: Literal['dense', 'nufft'] = 'nufft',
        origin: Float[Array, ' 2'] | None = None,
    ) -> Self:
        """Build the operator from boresight and detector quaternions.

        Args:
            index: Integer wavevector of each mode, shape (n_modes, 2).
            cell: The cell edges as columns, shape (2, 2), or the side of a square cell.
            qbore: Boresight quaternions in the horizon frame, shape (n_samples,).
            qdet: Detector offset quaternions, shape (n_detectors,).
            times: Time of each sample, shape (n_samples,).
            omega: Drift rate of the layer, minus the wind velocity over height, shape
                (2,).
            backend: `'nufft'` or `'dense'`.
            origin: The cell corner, zero by default.
        """
        cell = jnp.asarray(cell) * jnp.eye(2) if jnp.ndim(cell) == 0 else jnp.asarray(cell)
        index = jnp.asarray(index)
        n1, n2 = np.abs(np.asarray(index)).max(0)
        return cls(
            index=index,
            sampler=QuaternionSampler(kernel=SamplingKernel(), qbore=qbore, qdet=qdet),
            times=times,
            omega=omega,
            cell=cell,
            origin=jnp.zeros(2) if origin is None else jnp.asarray(origin),
            n_max=(int(n1), int(n2)),
            backend=backend,
            in_structure=jax.ShapeDtypeStruct((2, len(index)), times.dtype),
        )

    @property
    def modes(self) -> Float[Array, 'modes 2']:
        r"""The wavevector $\mathbf{q}_{\mathbf{n}}$ of each mode, in inverse screen units."""
        return 2 * jnp.pi * self.index @ jnp.linalg.inv(self.cell)

    def mv(self, x: Float[Array, '2 modes']) -> Float[Array, 'det samp']:
        if self.backend == 'nufft':
            return self._mv_nufft(x)

        def one_detector(idet):
            xi, void = self._cell_positions(idet)
            phase = 2 * jnp.pi * xi @ self.index.T
            return jnp.where(void, 0, jnp.cos(phase) @ x[0] + jnp.sin(phase) @ x[1])

        return jax.lax.map(one_detector, jnp.arange(self.sampler.shape[0]))

    def as_matrix(self) -> Float[Array, 'detsamp modes2']:
        r"""The dense design matrix $[\cos \mid \sin]$, rows ordered as the flattened samples."""
        xi, void = jax.vmap(self._cell_positions)(jnp.arange(self.sampler.shape[0]))
        phase = 2 * jnp.pi * xi @ self.index.T
        basis = jnp.where(void[..., None], 0, jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], -1))
        return basis.reshape(-1, 2 * self.index.shape[0])

    def _mv_nufft(self, x: Float[Array, '2 modes']) -> Float[Array, 'det samp']:
        # cos(q.y) c + sin(q.y) s = Re[(c - i s) e^{i q.y}]: one-sided grid, real part taken after.
        n1, n2 = self.n_max
        grid = jnp.zeros((2 * n1 + 1, 2 * n2 + 1), complex).at[
            self.index[:, 0] + n1, self.index[:, 1] + n2
        ]
        grid = grid.set(x[0] - 1j * x[1], unique_indices=True)
        # shortcut: one NUFFT over every sample at once; chunk over detectors at large sizes.
        xi, void = jax.vmap(self._cell_positions)(jnp.arange(self.sampler.shape[0]))
        points = 2 * jnp.pi * jnp.where(void[..., None], 0, xi)
        tod = nufft2(grid, points[..., 0].ravel(), points[..., 1].ravel(), iflag=1, eps=self.eps)
        return jnp.where(void, 0, jnp.real(tod).reshape(void.shape))

    def _cell_positions(
        self, idet: Int[Array, '']
    ) -> tuple[Float[Array, 'samp 2'], Bool[Array, ' samp']]:
        """Cell coordinates of one detector's drifted samples, and which have no pointing."""
        isamp = jnp.arange(self.sampler.shape[1])
        quats = self.sampler.quaternions((idet, isamp))
        void = jnp.all(quats.wxyz == 0, axis=-1)
        v = quats.rotate_vector(ZAXIS)
        u = v[..., :2] / jnp.where(void, 1, v[..., 2])[..., None]
        y = u + self.times[:, None] * self.omega
        return (y - self.origin) @ jnp.linalg.inv(self.cell).T, void


def cell_index(cell: Float[np.ndarray, '2 2'], q_max: float) -> Int[np.ndarray, 'modes 2']:
    r"""The integer wavevectors of a cell within a band limit.

    Returns every $\mathbf{n} \neq 0$ with $|2\pi B^{-\mathsf{T}}\mathbf{n}| \le q_\max$, one per
    $\pm\mathbf{n}$ pair. The constant mode is excluded, being degenerate with detector offsets.

    Args:
        cell: The cell edges $B$ as columns, shape (2, 2).
        q_max: The band limit, in inverse screen units.
    """
    cell = np.asarray(cell)
    # n_j = cell[:, j] . q / 2 pi, so |n_j| <= |cell[:, j]| q_max / 2 pi
    n1, n2 = (np.linalg.norm(cell, axis=0) * q_max / (2 * np.pi)).astype(int)
    nx, ny = np.meshgrid(np.arange(-n1, n1 + 1), np.arange(0, n2 + 1))
    n = np.stack([nx.ravel(), ny.ravel()], -1)
    q = 2 * np.pi * n @ np.linalg.inv(cell)
    keep = ((n[:, 1] > 0) | (n[:, 0] > 0)) & (np.linalg.norm(q, axis=1) <= q_max)
    return n[keep]


def fit_cell(
    u: Float[np.ndarray, 'det samp 2'],
    times: Float[np.ndarray, ' samp'],
    omega: Float[np.ndarray, ' 2'],
    margin: float,
) -> tuple[Float[np.ndarray, '2 2'], Float[np.ndarray, ' 2']]:
    """The parallelogram a layer's samples sweep, grown by a margin on every side.

    One edge follows the scan strip (the long axis of the lines of sight), the other the drift,
    or the strip's normal when the drift runs almost along the strip. The margin keeps the
    periodic field from tying opposite edges of the data together.

    Args:
        u: Screen positions of the samples before drift, shape (n_detectors, n_samples, 2).
        times: Time of each sample, shape (n_samples,).
        omega: Drift rate of the layer, shape (2,).
        margin: Margin around the swept region, in screen units.

    Returns:
        The `cell` and `origin` of a [`FourierScreenOperator`][furax.obs.atmosphere.FourierScreenOperator].
    """
    u, times, omega = np.asarray(u), np.asarray(times), np.asarray(omega)
    y = (u + times[:, None] * omega).reshape(-1, 2)
    e1 = np.linalg.eigh(np.cov(u.reshape(-1, 2).T))[1][:, -1]
    e2 = omega / max(np.linalg.norm(omega), 1e-300)
    if abs(e1[0] * e2[1] - e1[1] * e2[0]) < 0.3:
        e2 = np.array([-e1[1], e1[0]])
    edges = np.stack([e1, e2], axis=1)
    c = y @ np.linalg.inv(edges).T  # oblique coordinates of every sample
    # the margin is measured perpendicular to the opposite edge
    sin = abs(e1[0] * e2[1] - e1[1] * e2[0])
    lo, hi = c.min(0) - margin / sin, c.max(0) + margin / sin
    return edges * (hi - lo), edges @ lo
