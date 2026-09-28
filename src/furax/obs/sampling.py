import dataclasses
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Self, cast, dataclass_transform

import jax
import jax.numpy as jnp
import numpy as np
from fastquat import Quaternion
from jaxtyping import Array, Float, Int, Integer

from furax.core.utils import register_dataclass_with_keys
from furax.obs.coords import (
    XiEtaAngles,
    ZSPhi,
    gamma_angle_cos_sin,
    polarization_angle_cos_sin,
)
from furax.obs.landscapes import StokesLandscape
from furax.obs.operators import Spin2Rotation
from furax.obs.spin2 import transport_rotation
from furax.obs.stencil import Interpolation, Stencil
from furax.obs.stokes import Stokes, ValidStokesLiteral

__all__ = [
    'AbstractSampler',
    'AngleSampler',
    'DiscretizedBeam',
    'PolarizationFrame',
    'PrecomputedSampler',
    'SamplingKernel',
    'QuaternionSampler',
    'RotatedSampler',
    'PointingRows',
    'SampleIndex',
]


type PolarizationFrame = Literal['boresight', 'detector', 'sky']
"""The basis a [`QuaternionSampler`][furax.obs.sampling.QuaternionSampler] returns Q and U in."""

type SampleIndex = tuple[Int[Array, '...'], ...]
"""A batch of samples, as one integer array per axis of the samples' shape."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiscretizedBeam:
    r"""A detector beam, discretized as weighted nodes around the beam centre.

    A sample is the weighted sum of the map read at each node:

    $$d = \sum_k w_k \, m(\hat{n}_k),$$

    where $\hat{n}_k$ is the detector's line of sight, the beam centre, rotated by node $k$.

    The co-polar direction at a node is the beam-centre basis parallel-transported to the node.
    Accordingly, the $(Q, U)$ read at every node is transported to the beam centre, and the
    detector's polarization angle is applied once, at the centre.

    Attributes:
        nodes: Rotations from the beam centre to each node, in the detector frame, shape
            (n_nodes,) for the same nodes on every detector, or (n_detectors, n_nodes). Only the
            direction each rotation points the line of sight to is used, not its roll.
        weights: The weight of each node, shape (n_nodes,), shared by every Stokes component, or
            a [`Stokes`][] of the map's components, each of shape (n_nodes,), for a beam per
            component.
    """

    nodes: Quaternion
    weights: Float[Array, ' n_nodes'] | Stokes

    @classmethod
    def create(cls, nodes: Quaternion, weights: Float[Array, ' n_nodes'] | Stokes) -> Self:
        """Build a beam, checking that the nodes and weights agree.

        Args:
            nodes: Rotations from the beam centre to each node, see [`DiscretizedBeam`][].
            weights: The weight of each node, see [`DiscretizedBeam`][].
        """
        if len(nodes.shape) not in (1, 2):
            raise ValueError(
                f'beam nodes have shape {nodes.shape}, expected (n_nodes,) or '
                '(n_detectors, n_nodes)'
            )
        n_nodes = nodes.shape[-1]
        if not isinstance(weights, Stokes):
            weights = jnp.asarray(weights)
            if weights.ndim == 2 and weights.shape[1] == n_nodes:
                raise ValueError(
                    f'beam weights have shape {weights.shape}, expected ({n_nodes},); give one '
                    'set per Stokes component as a Stokes, e.g. StokesIQU(i=..., q=..., u=...)'
                )
        if weights.shape != (n_nodes,):
            per_component = ' per component' if isinstance(weights, Stokes) else ''
            raise ValueError(
                f'beam weights have shape {weights.shape}{per_component}, expected ({n_nodes},) '
                f'for {n_nodes} nodes'
            )
        return cls(nodes, weights)

    @classmethod
    def from_directions(
        cls,
        directions: Float[Array, '*detectors n_nodes 3'],
        weights: Float[Array, ' n_nodes'] | Stokes,
    ) -> Self:
        r"""Build a beam from the direction of each node, in the frame of a beam map.

        A beam map gives the beam at offsets $(\alpha, \delta)$ in right ascension and declination
        from the beam centre. The node at offset $(\alpha, \delta)$ has the direction
        $(\cos\delta \cos\alpha, \cos\delta \sin\alpha, \sin\delta)$, so the beam centre is along
        $+x$. When the detector's polarization angle is zero, $+\delta$ points north and
        $+\alpha$ east: the detector is sensitive to polarization along $-\delta$, and the beam
        turns with the detector.

        The nodes may be any set of directions within 90 degrees of the centre, e.g. the pixels of
        a gridded beam map or the centroids of a clustered one. The beam models the response to
        directions around the line of sight; it does not model the motion of the pointing during a
        sample.

        Args:
            directions: The direction of each node, shape (n_nodes, 3), or (n_detectors, n_nodes, 3)
                for a beam per detector. Directions are normalized to unit length.
            weights: The beam integrated over the solid angle of each node, e.g. $B \cos\delta$ on
                a map with equal steps in $\alpha$ and $\delta$, shared or per Stokes component as
                in [`DiscretizedBeam`][]. They are used as given.

        Examples:
            A beam of two nodes, 0.01 radians north and east of the centre:

            >>> import jax.numpy as jnp
            >>> directions = jnp.array([[jnp.cos(0.01), 0.0, jnp.sin(0.01)],
            ...                         [jnp.cos(0.01), jnp.sin(0.01), 0.0]])
            >>> beam = DiscretizedBeam.from_directions(directions, jnp.array([0.5, 0.5]))
            >>> beam.nodes.shape
            (2,)
        """
        directions = jnp.asarray(directions)
        if directions.ndim not in (2, 3) or directions.shape[-1] != 3:
            raise ValueError(
                f'beam directions have shape {directions.shape}, expected (n_nodes, 3) or '
                '(n_detectors, n_nodes, 3)'
            )
        unit = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
        y, z = unit[..., 1], unit[..., 2]
        # The detector looks along its z axis and is sensitive to polarization along its x axis,
        # which the beam map's -delta axis (-z) maps to; +alpha (+y) maps to its y axis. So the
        # detector-frame direction is (-z, y, x), whose orthographic coordinates are
        # (xi, eta) = (-y, z).
        nodes = XiEtaAngles(-y, z, jnp.zeros_like(y)).to_quaternion()
        return cls.create(nodes, weights)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Self:
        """Read a beam from a `.npz` file.

        The file holds the arrays:

        - `nodes`: the rotation of each node as a quaternion, scalar first, shape (n_nodes, 4), or
          (n_detectors, n_nodes, 4) for a beam per detector, see [`DiscretizedBeam`][].
        - `weights`: the weight of each node, shape (n_nodes,), or (n_components, n_nodes) for
          one set of weights per Stokes component.
        - `stokes`, optional: the Stokes components of per-component weights, e.g. `'IQU'`, one
          per row of `weights`. Empty or absent for weights shared by every component.

        Other arrays in the file are ignored.

        Args:
            path: The file to read.
        """
        with np.load(path) as data:
            nodes = Quaternion.from_array(jnp.asarray(data['nodes']))
            weights = jnp.asarray(data['weights'])
            stokes = str(data['stokes']) if 'stokes' in data.files else ''
        if stokes:
            if weights.ndim != 2 or weights.shape[0] != len(stokes):
                raise ValueError(
                    f'beam weights have shape {weights.shape}, expected one row per Stokes '
                    f'component of {stokes!r}'
                )
            # `class_for` rejects a string that is not a valid Stokes combination
            stokes_cls = Stokes.class_for(cast(ValidStokesLiteral, stokes))
            return cls.create(nodes, stokes_cls.from_array(weights))
        return cls.create(nodes, weights)

    def checked(self, landscape: StokesLandscape, n_detectors: int) -> Self:
        """The beam, checked against a map and detectors, with its weights in the map's dtype.

        Args:
            landscape: The map the beam reads.
            n_detectors: The number of detectors, which per-detector nodes must match.
        """
        if len(self.nodes.shape) == 2 and self.nodes.shape[0] != n_detectors:
            raise ValueError(
                f'beam nodes have shape {self.nodes.shape}, expected (n_nodes,) or '
                f'({n_detectors}, n_nodes) for {n_detectors} detectors'
            )
        weights = self.weights
        if isinstance(weights, Stokes):
            if weights.stokes != landscape.stokes:
                raise ValueError(
                    f'beam weights have Stokes components {weights.stokes!r}, expected those of '
                    f'the landscape, {landscape.stokes!r}'
                )
            weights = type(weights).from_array(jnp.asarray(weights.data, landscape.dtype))
        else:
            weights = jnp.asarray(weights, landscape.dtype)
        return dataclasses.replace(self, weights=weights)

    def nodes_for(self, idet: Int[Array, '...']) -> Quaternion:
        """The nodes of the given detectors, shape `(*idet.shape, n_nodes)`."""
        if len(self.nodes.shape) == 1:
            # shared by every detector
            shape = (*idet.shape, *self.nodes.wxyz.shape)
            return Quaternion.from_array(jnp.broadcast_to(self.nodes.wxyz, shape))
        return self.nodes[idet]

    def integrate(self, stencil: Stencil) -> Stencil:
        """Fold stencils `(..., n_nodes, neighbors)` into `(..., n_nodes * neighbors)`."""
        weights = self.weights.data if isinstance(self.weights, Stokes) else self.weights
        return stencil.integrated(weights)

    def intensity_only(self) -> Self:
        """Reduce per-Stokes weights to those of I, or to their mean when the map has no I."""
        weights = self.weights
        if not isinstance(weights, Stokes):
            return self
        reduced = weights.i if 'I' in weights.stokes else weights.data.mean(0)
        return dataclasses.replace(self, weights=reduced)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SamplingKernel:
    r"""How one sample reads the map: the interpolation, and the beam, if any.

    Without a beam, a sample reads the map along its line of sight. With a
    [`DiscretizedBeam`][], it reads it at every node of the beam. Each read is interpolated
    independently of the others: with bilinear interpolation, every node reads its four nearest
    pixels, so a beam of $K$ nodes reads up to $4K$ pixels per sample.

    Attributes:
        interpolation: How the map is read along a line of sight.
        beam: The beam each sample integrates over, or `None` to read the line of sight alone.
    """

    interpolation: Interpolation = field(default=Interpolation.NEAREST, metadata={'static': True})
    beam: DiscretizedBeam | None = None

    @property
    def reads_one_pixel(self) -> bool:
        """Whether a sample reads a single pixel: nearest neighbour, without a beam."""
        return self.interpolation is Interpolation.NEAREST and self.beam is None

    @property
    def weighs_per_stokes(self) -> bool:
        """Whether the beam weights differ between Stokes components."""
        return self.beam is not None and isinstance(self.beam.weights, Stokes)

    def intensity_only(self) -> Self:
        """The kernel reading the intensity alone, see [`DiscretizedBeam.intensity_only`][]."""
        if self.beam is None:
            return self
        return dataclasses.replace(self, beam=self.beam.intensity_only())


class PointingRows(NamedTuple):
    r"""How a batch of samples reads a map: one sparse row of the pointing matrix per sample.

    Sample $s$ reads $R(\psi_s) \sum_n w_{sn} R(\alpha_{sn}) m_{p_{sn}}$: the pixels $p$ and
    weights $w$ of the stencil, each neighbour's $(Q, U)$ rotated by its own angle $\alpha$, then
    the sum rotated by $\psi$ into the frame the sample is returned in. For a sampler on the
    sphere, $\alpha$ is the parallel transport to the line of sight, see
    [`transport_rotation`][furax.obs.spin2.transport_rotation], and $\psi$ the polarization angle.
    When the beam weights differ between Stokes components, they act on the rotated $Q$ and $U$,
    so $\psi$ is folded into every $\alpha$ instead and `polarization_rotation` is `None`.

    Attributes:
        stencil: The pixels each sample reads and their weights.
        neighbour_rotation: The rotation by $\alpha$ of each neighbour, or `None` when the map
            read has no polarization.
        polarization_rotation: The rotation by $\psi$ of each sample, or `None` to return the sum
            as it is.
    """

    stencil: Stencil
    neighbour_rotation: Spin2Rotation | None
    polarization_rotation: Spin2Rotation | None = None

    @classmethod
    def transported(
        cls,
        stencil: Stencil,
        line_of_sight: ZSPhi,
        rotation: Spin2Rotation | None,
        *,
        rotate_neighbours: bool = False,
    ) -> Self:
        r"""The rows of a stencil on the sphere, each neighbour transported to the line of sight.

        Args:
            stencil: The pixels each sample reads, with their positions.
            line_of_sight: The direction each sample is transported to.
            rotation: The rotation by $\psi$ into the frame the sample is returned in, or `None`.
            rotate_neighbours: Fold $\psi$ into every neighbour's rotation instead of applying it
                to the sum, for weights that differ between Stokes components.
        """
        if rotation is None:
            return cls(stencil, transport_rotation(stencil, line_of_sight))
        if rotate_neighbours:
            return cls(stencil, transport_rotation(stencil, line_of_sight, rotation))
        return cls(stencil, transport_rotation(stencil, line_of_sight), rotation)


@dataclass(frozen=True, kw_only=True)
@dataclass_transform(frozen_default=True, kw_only_default=True, field_specifiers=(field,))
class AbstractSampler(ABC):
    """Where each sample of a timestream, or of any array of samples, reads a map.

    A sampler has a [`shape`][furax.obs.sampling.AbstractSampler.shape], the shape of the samples
    it produces, and answers for any batch of them, given as a [`SampleIndex`][]. Nothing else
    about the shape is assumed: a timestream has shape (n_detectors, n_samples), but a sampler may
    as well read a map at a list of points, or at the pixels of another map. A batch may hold
    several detectors, or part of one's samples.

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
        """The shape of the samples."""

    def every_sample(self) -> SampleIndex:
        """The [`SampleIndex`][] of all the samples, in a batch of the samples' shape."""
        return tuple(jnp.indices(self.shape, sparse=True))

    @abstractmethod
    def pointing_rows(self, landscape: StokesLandscape, index: SampleIndex) -> PointingRows:
        """How a batch of samples reads the map.

        Args:
            landscape: The map being read.
            index: The samples of the batch.

        Returns:
            The pointing rows, of the shape of the batch, with a rotation when the map is
            polarized.
        """

    def nearest_indices(
        self, landscape: StokesLandscape, index: SampleIndex
    ) -> Integer[Array, '...']:
        """The pixel each sample of a batch reads, for a kernel that reads a single one.

        A shortcut for reading a map with no polarization, which needs the pixel index alone, not
        a stencil. It is the pixel of the
        [`pointing_rows`][furax.obs.sampling.AbstractSampler.pointing_rows] stencil, negative for a
        sample outside the map. It is only called when `kernel.reads_one_pixel`; the default takes
        it from the stencil.
        """
        stencil = self.pointing_rows(landscape, index).stencil
        # the stencil reads pixel 0 with zero weight for a sample outside the map
        return jnp.where(stencil.weights[..., 0] > 0, stencil.indices[..., 0], -1)

    def scaling(self, index: SampleIndex) -> Float[Array, '...'] | None:
        """A factor multiplying each sample of a batch, or `None` (the default) for none.

        It is applied after the gather and before the scatter. It is diagonal, so the operator and
        its transpose apply the same factor.
        """
        return None

    def with_kernel(self, kernel: SamplingKernel) -> 'AbstractSampler':
        """The same sampler with another kernel, e.g. to read the intensity alone."""
        return dataclasses.replace(self, kernel=kernel)

    def rotated(self, rotation: Spin2Rotation) -> 'AbstractSampler':
        r"""The same sampler, with the polarization of every sample rotated further by $\beta$.

        Args:
            rotation: The rotation by $\beta$, broadcastable to the shape of the samples.
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
        kernel: How each sample reads the map. Beam nodes compose with `qdet`; per-Stokes beam
            weights act on Q and U in the `frame` basis.
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

    def quaternions(self, index: SampleIndex | None = None) -> Quaternion:
        """The pointing of a batch of samples, or by default of every sample."""
        idet, isamp = self.every_sample() if index is None else index
        return self.qbore[isamp] * self.qdet[idet]

    def pointing_rows(self, landscape: StokesLandscape, index: SampleIndex) -> PointingRows:
        quats = self.quaternions(index)
        beam = self.kernel.beam
        if beam is None:
            stencil = self._stencil(landscape, quats)
        else:
            # Every node has its own stencil, and they fold into one stencil per sample. The
            # polarization of every pixel read is still transported to the line of sight, the
            # beam centre, so it is all in the same basis before the sum.
            # (*batch, 1) x (*batch, n_nodes) -> (*batch, n_nodes)
            node_quats = quats[..., None] * beam.nodes_for(index[0])
            stencil = beam.integrate(self._stencil(landscape, node_quats))
        if not landscape.has_spin2:
            return PointingRows(stencil, None)
        rotation = self._frame_rotation(quats, index[0])
        line_of_sight = landscape.quat2direction(quats)
        return PointingRows.transported(
            stencil, line_of_sight, rotation, rotate_neighbours=self.kernel.weighs_per_stokes
        )

    def nearest_indices(
        self, landscape: StokesLandscape, index: SampleIndex
    ) -> Integer[Array, '...']:
        return landscape.quat2index(self.quaternions(index))

    def to_angles(self, landscape: StokesLandscape) -> 'AngleSampler':
        """The same sampler, from the world angles of every sample and beam node, computed once.

        Bilinear interpolation only: a nearest-neighbour sampler finds its pixels with
        `quat2index`, and the angles of a sample on a pixel boundary can fall in a neighbour.
        """
        if self.kernel.interpolation is Interpolation.NEAREST:
            raise ValueError(
                'a nearest-neighbour sampler cannot be stored as angles, which may not find the '
                'same pixels: store its rows instead'
            )
        idet = self.every_sample()[0]
        quats = self.quaternions()
        theta, phi = landscape.quat2world(quats)
        rotation = self._frame_rotation(quats, idet)
        node_theta = node_phi = None
        if self.kernel.beam is not None:
            nodes = self.kernel.beam.nodes_for(idet)
            node_theta, node_phi = landscape.quat2world(quats[..., None] * nodes)
        return AngleSampler(
            kernel=self.kernel,
            theta=theta,
            phi=phi,
            polarization_rotation=rotation,
            node_theta=node_theta,
            node_phi=node_phi,
        )

    def _frame_rotation(self, quats: Quaternion, idet: Int[Array, '...']) -> Spin2Rotation | None:
        """The rotation from the meridian basis to the frame, if any."""
        if self.frame == 'sky':
            return None
        psi = Spin2Rotation.from_cos_sin(*polarization_angle_cos_sin(quats))
        if self.frame == 'detector':
            return psi
        # psi - gamma, with gamma the angle of the detector about the boresight
        gamma = Spin2Rotation.from_cos_sin(*gamma_angle_cos_sin(self.qdet[idet]))
        return psi.compose(gamma.inverse())

    def _stencil(self, landscape: StokesLandscape, quats: Quaternion) -> Stencil:
        """Read the map around each pointing, with the kernel's interpolation."""
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
        kernel: How each sample reads the map. With a beam, its nodes are given by `node_theta`
            and `node_phi`.
        theta: Co-latitude of every sample, in radians.
        phi: Longitude of every sample, in radians.
        polarization_rotation: The rotation by $\psi$ of every sample, or `None` to return the
            meridian basis.
        node_theta: Co-latitude of every beam node, shape `(*shape, n_nodes)`, or `None`.
        node_phi: Longitude of every beam node, or `None`.
    """

    theta: Float[Array, '...']
    phi: Float[Array, '...']
    polarization_rotation: Spin2Rotation | None = None
    node_theta: Float[Array, '... n_nodes'] | None = None
    node_phi: Float[Array, '... n_nodes'] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.theta.shape

    def pointing_rows(self, landscape: StokesLandscape, index: SampleIndex) -> PointingRows:
        theta, phi = self.theta[index], self.phi[index]
        beam = self.kernel.beam
        if beam is None:
            stencil = self._stencil(landscape, theta, phi)
        else:
            assert self.node_theta is not None and self.node_phi is not None
            node_stencil = self._stencil(landscape, self.node_theta[index], self.node_phi[index])
            stencil = beam.integrate(node_stencil)
        if not landscape.has_spin2:
            return PointingRows(stencil, None)
        rotation = self.polarization_rotation
        if rotation is not None:
            rotation = rotation[index]
        line_of_sight = ZSPhi.from_angles(theta, phi)
        return PointingRows.transported(
            stencil, line_of_sight, rotation, rotate_neighbours=self.kernel.weighs_per_stokes
        )

    def nearest_indices(
        self, landscape: StokesLandscape, index: SampleIndex
    ) -> Integer[Array, '...']:
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
    rotation for a map without polarization, and when every sample reads a single pixel, that
    pixel and a single rotation per sample, without weights. The rows are valid for the map they
    were computed for only.

    Attributes:
        kernel: The kernel of `source`.
        source: The sampler whose rows are stored, which still scales the samples (`scaling`).
        stencil: The stored stencils, or `None` when `nearest` is stored.
        neighbour_rotation: The stored rotation of each neighbour, or `None`. With `nearest`, the
            rotation of each sample, the polarization rotation folded in.
        polarization_rotation: The stored rotation of each sample, or `None`.
        nearest: The stored pixel of each sample, negative outside the map, when every sample
            reads a single pixel, or `None`.
    """

    source: AbstractSampler
    stencil: Stencil | None = None
    neighbour_rotation: Spin2Rotation | None = None
    polarization_rotation: Spin2Rotation | None = None
    nearest: Integer[Array, '...'] | None = None

    @classmethod
    def from_sampler(cls, sampler: AbstractSampler, landscape: StokesLandscape) -> Self:
        """Compute and store the pointing rows of a sampler for a map.

        Args:
            sampler: The sampler whose rows to store.
            landscape: The map the rows will read.
        """
        index = sampler.every_sample()
        if sampler.kernel.reads_one_pixel and not landscape.has_spin2:
            nearest = sampler.nearest_indices(landscape, index)
            return cls(kernel=sampler.kernel, source=sampler, nearest=nearest)
        pointing = sampler.pointing_rows(landscape, index)
        if sampler.kernel.reads_one_pixel:
            # A single neighbour, of weight one, or zero outside the map: store a negative pixel
            # for the zero weight, and the neighbour's transport composed with the polarization
            # rotation, rotations of one sample commuting.
            weights, indices = pointing.stencil.weights[..., 0], pointing.stencil.indices[..., 0]
            nearest = jnp.where(weights > 0, indices, -1)
            rotation = pointing.neighbour_rotation
            if rotation is not None:
                rotation = rotation[..., 0]
                if pointing.polarization_rotation is not None:
                    rotation = rotation.compose(pointing.polarization_rotation)
            return cls(
                kernel=sampler.kernel, source=sampler, nearest=nearest, neighbour_rotation=rotation
            )
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

    def pointing_rows(self, landscape: StokesLandscape, index: SampleIndex) -> PointingRows:
        if self.nearest is not None:
            indices = self.nearest[index]
            weights = jnp.ones((*indices.shape, 1), landscape.dtype)
            stencil = Stencil.unpositioned(indices[..., None], weights)
            rotation = self.neighbour_rotation
            return PointingRows(stencil, None if rotation is None else rotation[index][..., None])
        assert self.stencil is not None
        if self.stencil.weights.ndim > self.stencil.indices.ndim:
            # weights per Stokes component lead: the sample axes follow
            weights = self.stencil.weights[(slice(None), *index)]
            stencil = Stencil(self.stencil.indices[index], weights, None)
        else:
            stencil = Stencil(self.stencil.indices[index], self.stencil.weights[index], None)
        rotation = None if self.neighbour_rotation is None else self.neighbour_rotation[index]
        polarization = (
            None if self.polarization_rotation is None else self.polarization_rotation[index]
        )
        return PointingRows(stencil, rotation, polarization)

    def nearest_indices(
        self, landscape: StokesLandscape, index: SampleIndex
    ) -> Integer[Array, '...']:
        if self.nearest is None:
            return super().nearest_indices(landscape, index)
        return self.nearest[index]

    def scaling(self, index: SampleIndex) -> Float[Array, '...'] | None:
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
        rotation: The rotation by $\beta$, broadcastable to the shape of the samples.
    """

    source: AbstractSampler
    rotation: Spin2Rotation

    @property
    def shape(self) -> tuple[int, ...]:
        return self.source.shape

    def pointing_rows(self, landscape: StokesLandscape, index: SampleIndex) -> PointingRows:
        pointing = self.source.pointing_rows(landscape, index)
        if not landscape.has_spin2:
            return pointing
        rotation = self.rotation.broadcast_to(self.shape)[index]
        if pointing.polarization_rotation is not None:
            rotation = pointing.polarization_rotation.compose(rotation)
        return pointing._replace(polarization_rotation=rotation)

    def nearest_indices(
        self, landscape: StokesLandscape, index: SampleIndex
    ) -> Integer[Array, '...']:
        return self.source.nearest_indices(landscape, index)

    def scaling(self, index: SampleIndex) -> Float[Array, '...'] | None:
        return self.source.scaling(index)

    def with_kernel(self, kernel: SamplingKernel) -> AbstractSampler:
        source = self.source.with_kernel(kernel)
        return RotatedSampler(kernel=kernel, source=source, rotation=self.rotation)

    def rotated(self, rotation: Spin2Rotation) -> AbstractSampler:
        return self.source.rotated(self.rotation.compose(rotation))
