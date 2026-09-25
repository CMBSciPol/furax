import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Self, dataclass_transform

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jaxtyping import Array, Float, Int, Integer

from furax.core.utils import register_dataclass_with_keys
from furax.math.coords import to_polarization_angle_cos_sin
from furax.obs.landscapes import StokesLandscape
from furax.obs.spin2 import spin2_cos_sin_zs
from furax.obs.stencil import Interpolation, Stencil
from furax.obs.stokes import Stokes

__all__ = [
    'AbstractSampler',
    'SamplingKernel',
    'QuaternionSampler',
    'PointingRows',
]


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
    """The pixels a batch of samples reads, and the line of sight their polarization is carried to.

    Attributes:
        stencil: The pixels each sample reads and their weights.
        los_theta: Co-latitude of each sample's line of sight, which its polarization is
            transported to, in radians.
        los_phi: Longitude of the line of sight, in radians.
    """

    stencil: Stencil
    los_theta: Float[Array, ' *dims']
    los_phi: Float[Array, ' *dims']

    def transport_angles(self) -> Float[Array, ' *dims']:
        """The transport angle of a one-neighbour stencil, in radians."""
        assert self.stencil.positions is not None  # the caller transports, so it has positions
        cos_2delta, sin_2delta = spin2_cos_sin_zs(
            *self.stencil.positions,
            jnp.cos(self.los_theta)[..., None],
            jnp.sin(self.los_theta)[..., None],
            self.los_phi[..., None],
        )
        return 0.5 * jnp.arctan2(sin_2delta[..., 0], cos_2delta[..., 0])


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
        """The pixels a batch of samples reads, and where their polarization is transported.

        Args:
            landscape: The map being read.
            index: Indices into the first axis of the samples.

        Returns:
            The pointing rows, of shape `(batch, *shape[1:])`.
        """

    def nearest_indices(
        self, landscape: StokesLandscape, index: Int[Array, ' batch']
    ) -> Integer[Array, 'batch ...'] | None:
        """The pixel each sample reads, when it reads a single one, or `None`.

        A shortcut for reading a map with no polarization: when every sample reads one pixel, the
        gather needs its index alone, not a stencil. Samples outside the map are negative.
        `None` (the default) means the stencil of
        [`pointing_rows`][furax.obs.sampling.AbstractSampler.pointing_rows] must be used.
        """
        return None

    def polarization_rotation(
        self, index: Int[Array, ' batch']
    ) -> tuple[Float[Array, 'batch ...'], Float[Array, 'batch ...']] | None:
        r"""$(\cos\psi, \sin\psi)$ of the angle each sample's polarization is rotated by, or `None`.

        The sampled Q and U are first transported into the meridian basis of the line of sight
        given by [`pointing_rows`][furax.obs.sampling.AbstractSampler.pointing_rows], then
        rotated by $\psi$, e.g. into the frame of a detector. `None` (the default) leaves them in
        the meridian basis.
        """
        return None

    def scaling(self, index: Int[Array, ' batch']) -> Float[Array, 'batch ...'] | None:
        """A factor multiplying each sample of a batch, or `None` (the default) for none.

        It is applied after the gather and before the scatter. It is diagonal, so the operator and
        its transpose apply the same factor.
        """
        return None


class QuaternionSampler(AbstractSampler):
    """Detectors on a moving boresight, read along the pointing given by quaternions.

    The pointing of detector $d$ at sample $t$ is `qbore[t] * qdet[d]`, and the samples have shape
    (n_detectors, n_samples). The polarization is rotated into the frame of `qdet`.

    Attributes:
        kernel: What each sample integrates over. Its offsets, if any, compose with the
            pointing like a detector quaternion.
        qbore: Boresight quaternions, shape (n_samples,).
        qdet: Detector quaternions, shape (n_detectors,).
    """

    qbore: Quaternion
    qdet: Quaternion

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
            return self._read(landscape, quats)
        # Every read direction has its own stencil, and they fold into one stencil per
        # sample. The polarization of every pixel read is still transported to the line of
        # sight, so it is all in the same basis before the sum.
        theta, phi = landscape.quat2world(quats)
        # (batch, samp, 1) x (batch, 1, n_offsets) -> (batch, samp, n_offsets)
        offset_pointing = self._read(landscape, quats[:, :, None] * offsets[:, None, :])
        return PointingRows(self.kernel.integrate(offset_pointing.stencil), theta, phi)

    def nearest_indices(
        self, landscape: StokesLandscape, index: Int[Array, ' batch']
    ) -> Integer[Array, 'batch samp'] | None:
        if not self.kernel.reads_one_pixel:
            return None
        return landscape.quat2index(self.quaternions(index))

    def polarization_rotation(
        self, index: Int[Array, ' batch']
    ) -> tuple[Float[Array, 'batch samp'], Float[Array, 'batch samp']]:
        return to_polarization_angle_cos_sin(self.quaternions(index))

    def _read(self, landscape: StokesLandscape, quats: Quaternion) -> PointingRows:
        """Read the map around each direction, with the kernel's interpolation."""
        world = landscape.quat2world(quats)
        if self.kernel.interpolation is Interpolation.NEAREST:
            # index through `quat2index`, so that the pixel is the one the hit map counts
            stencil = landscape.index2stencil(landscape.quat2index(quats))
        else:
            stencil = landscape.world2stencil(*world, self.kernel.interpolation)
        return PointingRows(stencil, *world)
