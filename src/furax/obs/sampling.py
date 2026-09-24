import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Self, dataclass_transform

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jaxtyping import Array, Float, Int, Integer

from furax.core.utils import register_dataclass_with_keys
from furax.math.coords import gamma_angle_cos_sin, polarization_angle_cos_sin
from furax.obs.landscapes import StokesLandscape
from furax.obs.spin2 import transport_rotation
from furax.obs.stencil import Interpolation, Stencil
from furax.obs.stokes import Stokes

__all__ = [
    'AbstractSampler',
    'AngleSampler',
    'PolarizationFrame',
    'PrecomputedSampler',
    'SamplingKernel',
    'QuaternionSampler',
    'RotatedSampler',
    'PointingRows',
]


type PolarizationFrame = Literal['boresight', 'detector', 'sky']
"""The basis a [`QuaternionSampler`][furax.obs.sampling.QuaternionSampler] returns Q and U in."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SamplingKernel:
    r"""The integration kernel for one sample.

    A sample reads the map at one or more directions and returns their weighted sum:

    $$d = \sum_k w_k \, m(\hat{n}_k),$$

    where $\hat{n}_k$, a read direction, is the line of sight of the sample composed with
    offset $k$. Without offsets, the only read direction is the line of sight, with unit
    weight. The offsets model an integration within a sample: a finite time integration, a
    pixel window, or a beam.

    Each value $m(\hat{n}_k)$ is then read from the pixelized map with the kernel's interpolation,
    independently of the offsets: with bilinear interpolation, every direction reads its four
    nearest pixels, so a kernel of $K$ offsets reads up to $4K$ pixels per sample.

    Build one with [`SamplingKernel.create`][], which validates the offsets and weights against
    the map and the number of detectors; the constructor does not.

    Attributes:
        interpolation: How each direction is read from the map.
        offsets: Rotations from the line of sight to each read direction, shape (n_offsets,)
            for the same offsets on every detector, or (n_detectors, n_offsets). `None` to read
            the line of sight alone.
        weights: The non-negative weight of each offset, shape (n_offsets,), shared by every
            Stokes component, or a [`Stokes`][] of the map's components, each of shape
            (n_offsets,). `None` if `offsets` is `None`.
    """

    interpolation: Interpolation = field(default=Interpolation.NEAREST, metadata={'static': True})
    offsets: Quaternion | None = None
    weights: Float[Array, ' n_offsets'] | Stokes | None = None

    @classmethod
    def create(
        cls,
        landscape: StokesLandscape,
        n_detectors: int,
        *,
        interpolation: Interpolation = Interpolation.NEAREST,
        offsets: Quaternion | None = None,
        weights: Float[Array, ' n_offsets'] | Stokes | None = None,
    ) -> Self:
        """Build a kernel, checking the offsets and weights against the map and detectors.

        The weights are cast to the map's dtype. They are normalized to sum to one, per Stokes
        component, over the offsets whose pixels are in the map: a sample partly off a partial-sky
        map is renormalized to the part in view, like a partly covered bilinear sample.
        """
        if (offsets is None) != (weights is None):
            raise ValueError('offsets and their weights must be given together')
        if offsets is None:
            return cls(interpolation)
        assert weights is not None
        n_offsets = offsets.shape[-1]
        if offsets.shape not in {(n_offsets,), (n_detectors, n_offsets)}:
            raise ValueError(
                f'offsets has shape {offsets.shape}, expected ({n_offsets},) or '
                f'({n_detectors}, {n_offsets}) for {n_detectors} detectors'
            )
        return cls(interpolation, offsets, _checked_weights(weights, landscape, n_offsets))

    @property
    def reads_one_pixel(self) -> bool:
        """Whether a sample reads a single pixel: nearest neighbour, without offsets."""
        return self.interpolation is Interpolation.NEAREST and self.offsets is None

    def integrate(self, stencil: Stencil) -> Stencil:
        """Fold stencils of shape (..., n_offsets, neighbors) into (..., n_offsets * neighbors)."""
        assert self.weights is not None
        weights = self.weights.data if isinstance(self.weights, Stokes) else self.weights
        return stencil.integrated(weights)

    def offsets_for(self, idet: Int[Array, ' batch']) -> Quaternion | None:
        """Shape (batch, n_offsets), with shared offsets broadcast to every detector."""
        if self.offsets is None:
            return None
        if len(self.offsets.shape) == 1:
            # shared by every detector
            shape = (*idet.shape, *self.offsets.wxyz.shape)
            return Quaternion.from_array(jnp.broadcast_to(self.offsets.wxyz, shape))
        return self.offsets[idet]

    def intensity_only(self) -> Self:
        """Reduce per-Stokes weights to those of I, or to their mean when the map has no I."""
        weights = self.weights
        if not isinstance(weights, Stokes):
            return self
        reduced = weights.i if 'I' in weights.stokes else weights.data.mean(0)
        return dataclasses.replace(self, weights=reduced)


def _checked_weights(
    weights: Float[Array, ' n_offsets'] | Stokes, landscape: StokesLandscape, n_offsets: int
) -> Float[Array, ' n_offsets'] | Stokes:
    """The offset weights in the landscape's dtype, after checking they fit the offsets and map."""
    if isinstance(weights, Stokes):
        if weights.stokes != landscape.stokes:
            raise ValueError(
                f'offset weights have Stokes components {weights.stokes!r}, expected those of '
                f'the landscape, {landscape.stokes!r}'
            )
        if weights.shape != (n_offsets,):
            raise ValueError(
                f'offset weights have shape {weights.shape} per component, expected '
                f'({n_offsets},) for {n_offsets} offsets'
            )
        return type(weights).from_array(jnp.asarray(weights.data, dtype=landscape.dtype))
    weights = jnp.asarray(weights, dtype=landscape.dtype)
    if weights.shape != (n_offsets,):
        hint = (
            ', or give one set per Stokes component as a Stokes, '
            'e.g. StokesIQU(i=..., q=..., u=...)'
            if weights.ndim == 2
            else ''
        )
        raise ValueError(
            f'offset weights have shape {weights.shape}, expected ({n_offsets},) for '
            f'{n_offsets} offsets{hint}'
        )
    return weights


class PointingRows(NamedTuple):
    r"""How a batch of samples reads a map: one sparse row of the pointing matrix per sample.

    Sample $s$ reads $R(\psi_s) \sum_n w_{sn} R(\alpha_{sn}) m_{p_{sn}}$: the pixels $p$ and
    weights $w$ of the stencil, each neighbour's $(Q, U)$ rotated by its own angle $\alpha$, then
    the sum rotated by $\psi$ into the frame the sample is returned in. For a sampler on the
    sphere, $\alpha$ is the parallel transport to the line of sight, see
    [`transport_rotation`][furax.obs.spin2.transport_rotation], and $\psi$ the polarization angle.
    When the weights differ between Stokes components, they act on the rotated $Q$ and $U$, so
    $\psi$ is folded into every $\alpha$ instead and `polarization_rotation` is `None`.

    Attributes:
        stencil: The pixels each sample reads and their weights.
        neighbour_rotation: $(\cos 2\alpha, \sin 2\alpha)$ for each neighbour, or `None` when
            the map read has no polarization.
        polarization_rotation: $(\cos 2\psi, \sin 2\psi)$ for each sample, or `None` to return
            the sum as it is.
    """

    stencil: Stencil
    neighbour_rotation: tuple[Float[Array, '...'], Float[Array, '...']] | None
    polarization_rotation: tuple[Float[Array, '...'], Float[Array, '...']] | None = None


def _doubled(
    cos: Float[Array, '...'], sin: Float[Array, '...']
) -> tuple[Float[Array, '...'], Float[Array, '...']]:
    r"""$(\cos 2x, \sin 2x)$ from $(\cos x, \sin x)$, the form a rotation of Q and U takes."""
    return cos**2 - sin**2, 2 * cos * sin


def _composed(
    first: tuple[Float[Array, '...'], Float[Array, '...']],
    then: tuple[Float[Array, '...'], Float[Array, '...']],
) -> tuple[Float[Array, '...'], Float[Array, '...']]:
    """The (cos, sin) of the sum of two angles, from those of each."""
    (cos_a, sin_a), (cos_b, sin_b) = first, then
    return cos_a * cos_b - sin_a * sin_b, sin_a * cos_b + cos_a * sin_b


def _polarized(
    stencil: Stencil,
    theta: Float[Array, '...'],
    phi: Float[Array, '...'],
    rotation: tuple[Float[Array, '...'], Float[Array, '...']] | None,
    kernel: 'SamplingKernel',
) -> PointingRows:
    """The pointing rows of a polarized map, transported to the line of sight `(theta, phi)`."""
    if rotation is None:
        return PointingRows(stencil, transport_rotation(stencil, theta, phi))
    if isinstance(kernel.weights, Stokes):
        # weights per component act on the rotated Q and U: turn every neighbour before the sum
        return PointingRows(stencil, transport_rotation(stencil, theta, phi, rotation))
    return PointingRows(stencil, transport_rotation(stencil, theta, phi), rotation)


@dataclass(frozen=True, kw_only=True)
@dataclass_transform(frozen_default=True, kw_only_default=True, field_specifiers=(field,))
class AbstractSampler(ABC):
    """Where each sample of a timestream, or of any array of samples, reads a map.

    A sampler has a [`shape`][furax.obs.sampling.AbstractSampler.shape], the shape of the samples
    it produces, and answers for a batch of indices into its first axis. Nothing else about the
    shape is assumed: a timestream has shape (n_detectors, n_samples), but a sampler may as well
    read a map at a list of points, or at the pixels of another map.

    Attributes:
        kernel: What each sample integrates over.
    """

    kernel: SamplingKernel

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(frozen=True, kw_only=True)(cls)
        # unflattening bypasses `__init__`, so that a check in `__post_init__` runs on the
        # sampler a user builds, not on the placeholder leaves of a JAX transformation
        register_dataclass_with_keys(cls)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the samples. Batches index its first axis."""

    @abstractmethod
    def pointing_rows(
        self, landscape: StokesLandscape, index: Int[Array, ' batch']
    ) -> PointingRows:
        """How a batch of samples reads the map.

        Args:
            landscape: The map being read.
            index: Indices into the first axis of the samples.

        Returns:
            The pointing rows, of shape `(batch, *shape[1:])`, with a rotation when the map is
            polarized.
        """

    def nearest_indices(
        self, landscape: StokesLandscape, index: Int[Array, ' batch']
    ) -> Integer[Array, 'batch ...'] | None:
        """The pixel each sample reads, when it reads a single one, or `None`.

        A shortcut for reading a map with no polarization: when every sample reads one pixel, the
        gather needs its index alone, not a stencil. It must be the pixel of the
        [`pointing_rows`][furax.obs.sampling.AbstractSampler.pointing_rows] stencil; samples
        outside the map are negative. `None` (the default) means the stencil must be used.
        """
        return None

    def scaling(self, index: Int[Array, ' batch']) -> Float[Array, 'batch ...'] | None:
        """A factor multiplying each sample of a batch, or `None` (the default) for none.

        It is applied after the gather and before the scatter. It is diagonal, so the operator and
        its transpose apply the same factor.
        """
        return None

    def with_kernel(self, kernel: SamplingKernel) -> 'AbstractSampler':
        """The same sampler with another kernel, e.g. to read the intensity alone."""
        return dataclasses.replace(self, kernel=kernel)

    def rotated(
        self, rotation: tuple[Float[Array, '...'], Float[Array, '...']]
    ) -> 'AbstractSampler':
        r"""The same sampler, with the polarization of every sample rotated further by $\beta$.

        Args:
            rotation: $(\cos 2\beta, \sin 2\beta)$, broadcastable to the shape of the samples.
        """
        return RotatedSampler(kernel=self.kernel, source=self, rotation=rotation)


class QuaternionSampler(AbstractSampler):
    r"""Detectors on a moving boresight, read along the pointing given by quaternions.

    The pointing of detector $d$ at sample $t$ is `qbore[t] * qdet[d]`, and the samples have shape
    (n_detectors, n_samples). Q and U are returned in the meridian basis of the line of sight
    rotated by $\psi - \gamma$, with $\psi$ the polarization angle and $\gamma$ the detector's
    angle about the boresight (`'boresight'` frame, the default), by $\psi$ (`'detector'`), or not
    rotated (`'sky'`).

    Attributes:
        kernel: What each sample integrates over. Offsets compose with `qdet`; per-Stokes weights
            act on Q and U in the `frame` basis.
        qbore: Boresight quaternions, shape (n_samples,).
        qdet: Detector quaternions, shape (n_detectors,).
        frame: The basis Q and U are returned in.
    """

    qbore: Quaternion
    qdet: Quaternion
    frame: PolarizationFrame = field(default='boresight', metadata={'static': True})

    @property
    def shape(self) -> tuple[int, ...]:
        return self.qdet.shape[0], self.qbore.shape[0]

    def quaternions(self, index: Int[Array, ' batch'] | None = None) -> Quaternion:
        """The pointing of a batch of detectors, or of every detector, shape (batch, n_samples)."""
        qdet = self.qdet if index is None else self.qdet[index]
        return self.qbore * qdet[:, None]

    def pointing_rows(
        self, landscape: StokesLandscape, index: Int[Array, ' batch']
    ) -> PointingRows:
        quats = self.quaternions(index)
        offsets = self.kernel.offsets_for(index)
        if offsets is None:
            stencil = self._stencil(landscape, quats)
        else:
            # Every read direction has its own stencil, and they fold into one stencil per
            # sample. The polarization of every pixel read is still transported to the line of
            # sight, so it is all in the same basis before the sum.
            # (batch, samp, 1) x (batch, 1, n_offsets) -> (batch, samp, n_offsets)
            offset_quats = quats[:, :, None] * offsets[:, None, :]
            stencil = self.kernel.integrate(self._stencil(landscape, offset_quats))
        if not landscape.has_spin2:
            return PointingRows(stencil, None)
        rotation = self._frame_rotation(quats, index)
        theta, phi = landscape.quat2world(quats)
        return _polarized(stencil, theta, phi, rotation, self.kernel)

    def nearest_indices(
        self, landscape: StokesLandscape, index: Int[Array, ' batch']
    ) -> Integer[Array, 'batch samp'] | None:
        if not self.kernel.reads_one_pixel:
            return None
        return landscape.quat2index(self.quaternions(index))

    def to_angles(self, landscape: StokesLandscape) -> 'AngleSampler':
        """The same sampler, from the world angles of every direction, computed once."""
        index = jnp.arange(self.shape[0])
        quats = self.quaternions(index)
        theta, phi = landscape.quat2world(quats)
        rotation = self._frame_rotation(quats, index)
        offset_theta = offset_phi = None
        offsets = self.kernel.offsets_for(index)
        if offsets is not None:
            offset_theta, offset_phi = landscape.quat2world(quats[:, :, None] * offsets[:, None, :])
        return AngleSampler(
            kernel=self.kernel,
            theta=theta,
            phi=phi,
            polarization_rotation=rotation,
            offset_theta=offset_theta,
            offset_phi=offset_phi,
        )

    def _frame_rotation(
        self, quats: Quaternion, index: Int[Array, ' batch']
    ) -> tuple[Float[Array, 'batch samp'], Float[Array, 'batch samp']] | None:
        r"""$(\cos 2x, \sin 2x)$ of the angle $x$ from the meridian basis to the frame, if any."""
        if self.frame == 'sky':
            return None
        psi = polarization_angle_cos_sin(quats)
        if self.frame == 'detector':
            return _doubled(*psi)
        # psi - gamma, with gamma the angle of the detector about the boresight
        cos_gamma, sin_gamma = gamma_angle_cos_sin(self.qdet[index])
        cos_gamma, sin_gamma = cos_gamma[:, None], sin_gamma[:, None]
        return _doubled(*_composed(psi, (cos_gamma, -sin_gamma)))

    def _stencil(self, landscape: StokesLandscape, quats: Quaternion) -> Stencil:
        """Read the map around each direction, with the kernel's interpolation."""
        if self.kernel.interpolation is Interpolation.NEAREST:
            # index through `quat2index`, so that the pixel is the one the hit map counts
            return landscape.index2stencil(landscape.quat2index(quats))
        return landscape.world2stencil(*landscape.quat2world(quats), self.kernel.interpolation)


class AngleSampler(AbstractSampler):
    r"""Samples at given world angles, with a polarization angle each.

    The lines of sight are the co-latitude $\theta$ and longitude $\phi$ of each sample, and the
    polarization is returned in a frame rotated by $\psi$ from the meridian basis of that direction.
    It holds the pointing as arrays, which costs memory but no trigonometry on every apply:
    [`QuaternionSampler.to_angles`][furax.obs.sampling.QuaternionSampler.to_angles] builds one.

    Attributes:
        kernel: What each sample integrates over. With offsets, their directions are given by
            `offset_theta` and `offset_phi`.
        theta: Co-latitude of every sample, in radians.
        phi: Longitude of every sample, in radians.
        polarization_rotation: $(\cos 2\psi, \sin 2\psi)$ of every sample, or `None` to return
            the meridian basis.
        offset_theta: Co-latitude of every read direction, shape `(*shape, n_offsets)`, or `None`.
        offset_phi: Longitude of every read direction, or `None`.
    """

    theta: Float[Array, '...']
    phi: Float[Array, '...']
    polarization_rotation: tuple[Float[Array, '...'], Float[Array, '...']] | None = None
    offset_theta: Float[Array, '... n_offsets'] | None = None
    offset_phi: Float[Array, '... n_offsets'] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.theta.shape

    def pointing_rows(
        self, landscape: StokesLandscape, index: Int[Array, ' batch']
    ) -> PointingRows:
        theta, phi = self.theta[index], self.phi[index]
        if self.offset_theta is None:
            stencil = self._stencil(landscape, theta, phi)
        else:
            assert self.offset_phi is not None
            offset_stencil = self._stencil(
                landscape, self.offset_theta[index], self.offset_phi[index]
            )
            stencil = self.kernel.integrate(offset_stencil)
        if not landscape.has_spin2:
            return PointingRows(stencil, None)
        rotation = self.polarization_rotation
        if rotation is not None:
            rotation = rotation[0][index], rotation[1][index]
        return _polarized(stencil, theta, phi, rotation, self.kernel)

    def nearest_indices(
        self, landscape: StokesLandscape, index: Int[Array, ' batch']
    ) -> Integer[Array, 'batch ...'] | None:
        if not self.kernel.reads_one_pixel:
            return None
        return landscape.world2index(self.theta[index], self.phi[index])

    def _stencil(
        self, landscape: StokesLandscape, theta: Float[Array, '...'], phi: Float[Array, '...']
    ) -> Stencil:
        if self.kernel.interpolation is Interpolation.NEAREST:
            return landscape.index2stencil(landscape.world2index(theta, phi))
        return landscape.world2stencil(theta, phi, self.kernel.interpolation)


class PrecomputedSampler(AbstractSampler):
    """Another sampler, with the rows of its pointing matrix computed once for a given map.

    Reading a map through it is a gather of stored indices, weights and rotations: the fastest
    apply, at the cost of storing them, per neighbour. It stores only what the map needs: no
    rotation for a map without polarization, and the pixel index alone when every sample reads a
    single pixel of such a map. The rows are valid for the map they were computed for only.

    Attributes:
        kernel: The kernel of `source`.
        source: The sampler whose rows are stored, which still scales the samples (`scaling`).
        stencil: The stored stencils, or `None` when `nearest` suffices.
        neighbour_rotation: The stored rotation of each neighbour, or `None`.
        polarization_rotation: The stored rotation of each sample, or `None`.
        nearest: The stored pixel of each sample, when every sample reads a single pixel of a map
            without polarization, or `None`.
    """

    source: AbstractSampler
    stencil: Stencil | None = None
    neighbour_rotation: tuple[Float[Array, '...'], Float[Array, '...']] | None = None
    polarization_rotation: tuple[Float[Array, '...'], Float[Array, '...']] | None = None
    nearest: Integer[Array, '...'] | None = None

    @classmethod
    def from_sampler(cls, sampler: AbstractSampler, landscape: StokesLandscape) -> Self:
        """Compute and store the pointing rows of a sampler for a map.

        Args:
            sampler: The sampler whose rows to store.
            landscape: The map the rows will read.
        """
        index = jnp.arange(sampler.shape[0])
        nearest = sampler.nearest_indices(landscape, index)
        if nearest is not None and not landscape.has_spin2:
            return cls(kernel=sampler.kernel, source=sampler, nearest=nearest)
        pointing = sampler.pointing_rows(landscape, index)
        # the positions only served to compute the rotation, which is cached instead
        stencil = Stencil(pointing.stencil.indices, pointing.stencil.weights, None)
        return cls(
            kernel=sampler.kernel,
            source=sampler,
            stencil=stencil,
            neighbour_rotation=pointing.neighbour_rotation,
            polarization_rotation=pointing.polarization_rotation,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.source.shape

    def pointing_rows(
        self, landscape: StokesLandscape, index: Int[Array, ' batch']
    ) -> PointingRows:
        if self.stencil is None:
            assert self.nearest is not None
            indices = self.nearest[index]
            weights = jnp.ones((*indices.shape, 1), landscape.dtype)
            return PointingRows(Stencil.unpositioned(indices[..., None], weights), None)
        if self.stencil.weights.ndim > self.stencil.indices.ndim:
            # weights per Stokes component lead: the batch axis is the second one
            stencil = Stencil(self.stencil.indices[index], self.stencil.weights[:, index], None)
        else:
            stencil = Stencil(self.stencil.indices[index], self.stencil.weights[index], None)
        rotation = polarization = None
        if self.neighbour_rotation is not None:
            rotation = self.neighbour_rotation[0][index], self.neighbour_rotation[1][index]
        if self.polarization_rotation is not None:
            polarization = (
                self.polarization_rotation[0][index],
                self.polarization_rotation[1][index],
            )
        return PointingRows(stencil, rotation, polarization)

    def nearest_indices(
        self, landscape: StokesLandscape, index: Int[Array, ' batch']
    ) -> Integer[Array, 'batch ...'] | None:
        return None if self.nearest is None else self.nearest[index]

    def scaling(self, index: Int[Array, ' batch']) -> Float[Array, 'batch ...'] | None:
        return self.source.scaling(index)

    def with_kernel(self, kernel: SamplingKernel) -> AbstractSampler:
        """The source sampler with another kernel: the cache no longer applies."""
        return self.source.with_kernel(kernel)


class RotatedSampler(AbstractSampler):
    r"""Another sampler, with the polarization of every sample rotated further by $\beta$.

    What [`PointingOperator`][furax.obs.pointing.PointingOperator] builds when it absorbs a
    [`QURotationOperator`][furax.obs.operators.QURotationOperator] applied to its output, so
    that the two cost one pass over the samples instead of two.

    Attributes:
        kernel: The kernel of the rotated sampler.
        source: The sampler whose samples are rotated.
        rotation: $(\cos 2\beta, \sin 2\beta)$, broadcastable to the shape of the samples.
    """

    source: AbstractSampler
    rotation: tuple[Float[Array, '...'], Float[Array, '...']]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.source.shape

    def pointing_rows(
        self, landscape: StokesLandscape, index: Int[Array, ' batch']
    ) -> PointingRows:
        pointing = self.source.pointing_rows(landscape, index)
        if not landscape.has_spin2:
            return pointing
        cos_2b, sin_2b = (jnp.broadcast_to(r, self.shape)[index] for r in self.rotation)
        rotation = cos_2b, sin_2b
        if pointing.polarization_rotation is not None:
            rotation = _composed(pointing.polarization_rotation, rotation)
        return pointing._replace(polarization_rotation=rotation)

    def nearest_indices(
        self, landscape: StokesLandscape, index: Int[Array, ' batch']
    ) -> Integer[Array, 'batch ...'] | None:
        return self.source.nearest_indices(landscape, index)

    def scaling(self, index: Int[Array, ' batch']) -> Float[Array, 'batch ...'] | None:
        return self.source.scaling(index)

    def with_kernel(self, kernel: SamplingKernel) -> AbstractSampler:
        source = self.source.with_kernel(kernel)
        return RotatedSampler(kernel=kernel, source=source, rotation=self.rotation)

    def rotated(self, rotation: tuple[Float[Array, '...'], Float[Array, '...']]) -> AbstractSampler:
        return self.source.rotated(_composed(self.rotation, rotation))
