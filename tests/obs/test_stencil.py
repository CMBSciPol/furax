import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax.obs.stencil import Interpolation, SkyPositions, Stencil


def _positions(shape: tuple[int, ...]) -> SkyPositions:
    """Arbitrary but valid neighbour positions for a stencil of the given shape."""
    rng = np.random.default_rng(0)
    theta = jnp.asarray(rng.uniform(0.1, np.pi - 0.1, shape))
    phi = jnp.asarray(rng.uniform(0.0, 2 * np.pi, shape))
    return SkyPositions(jnp.cos(theta), jnp.sin(theta), phi)


class TestInterpolation:
    @pytest.mark.parametrize('interpolation', list(Interpolation))
    def test_the_value_is_the_neighbour_count(self, interpolation):
        """The value is how many pixels the sample reads, which a stencil can be checked against."""
        positions = _positions((3, interpolation))
        stencil = Stencil.resolve(
            jnp.zeros((3, interpolation), jnp.int32), jnp.ones((3, interpolation)), positions
        )
        assert stencil.n_neighbors == interpolation


class TestResolution:
    """Every constructor resolves what it builds; `Stencil.unpositioned` is the shortest one."""

    def test_normalizes_the_weights(self):
        stencil = Stencil.unpositioned(jnp.array([[0, 1, 2, 3]]), jnp.array([[1.0, 1.0, 1.0, 1.0]]))
        assert_allclose(np.asarray(stencil.weights), 0.25)

    def test_out_of_map_neighbours_are_dropped_and_the_rest_rescaled(self):
        stencil = Stencil.unpositioned(
            jnp.array([[0, -1, 2, 3]]), jnp.array([[0.4, 0.4, 0.1, 0.1]])
        )
        assert_array_equal(np.asarray(stencil.indices), [[0, 0, 2, 3]])
        assert_allclose(np.asarray(stencil.weights), [[2 / 3, 0.0, 1 / 6, 1 / 6]])

    def test_a_fully_uncovered_sample_is_zero_rather_than_nan(self):
        """Dividing by a zero weight sum must be guarded: such a sample contributes nothing."""
        stencil = Stencil.unpositioned(jnp.array([[-1, -1]]), jnp.array([[0.7, 0.3]]))
        assert_array_equal(np.asarray(stencil.indices), [[0, 0]])
        assert_array_equal(np.asarray(stencil.weights), [[0.0, 0.0]])


class TestStencil:
    def test_resolve_is_the_resolved_stencil(self):
        indices, weights = jnp.array([[0, -1, 2, 3]]), jnp.array([[0.4, 0.4, 0.1, 0.1]])
        positions = _positions((1, 4))
        stencil = Stencil.resolve(indices, weights, positions)

        assert_array_equal(np.asarray(stencil.indices), [[0, 0, 2, 3]])
        assert_allclose(np.asarray(stencil.weights), [[2 / 3, 0.0, 1 / 6, 1 / 6]])
        # a re-numbering never moves a pixel, so the positions are passed through untouched
        assert_array_equal(np.asarray(stencil.positions.z), np.asarray(positions.z))
        assert stencil.n_neighbors == 4

    def test_astype_casts_the_weights_and_positions_but_not_the_indices(self):
        positions = _positions((3, 4))
        stencil = Stencil.resolve(jnp.zeros((3, 4), jnp.int32), jnp.ones((3, 4)), positions)

        cast = stencil.astype(jnp.float32)
        assert cast.weights.dtype == jnp.float32
        assert all(component.dtype == jnp.float32 for component in cast.positions)
        assert cast.indices.dtype == jnp.int32

    def test_astype_leaves_a_stencil_without_positions_alone(self):
        stencil = Stencil.unpositioned(jnp.array([[0, 1]]), jnp.ones((1, 2))).astype(jnp.float32)
        assert stencil.weights.dtype == jnp.float32
        assert stencil.positions is None

    def test_nearest_holds_one_neighbour_of_unit_weight(self):
        theta, phi = jnp.array([0.3, 1.2]), jnp.array([0.0, 4.0])
        stencil = Stencil.nearest(jnp.array([7, 9]), theta, phi)

        assert stencil.n_neighbors == 1
        assert stencil.indices.shape == (2, 1)
        assert_array_equal(np.asarray(stencil.indices[..., 0]), [7, 9])
        assert_array_equal(np.asarray(stencil.weights), 1.0)
        assert_allclose(np.asarray(stencil.positions.z[..., 0]), np.cos(np.asarray(theta)))
        assert_allclose(np.asarray(stencil.positions.sth[..., 0]), np.sin(np.asarray(theta)))
        assert_array_equal(np.asarray(stencil.positions.phi[..., 0]), np.asarray(phi))

    def test_nearest_outside_the_map_contributes_nothing(self):
        stencil = Stencil.nearest(jnp.array([-1]), jnp.array([0.3]), jnp.array([0.0]))
        assert_array_equal(np.asarray(stencil.indices), [[0]])
        assert_array_equal(np.asarray(stencil.weights), [[0.0]])

    def test_reindexed_renormalizes_against_the_new_numbering(self):
        """Dropping a neighbour leaves the rest no longer summing to one."""
        positions = _positions((1, 4))
        stencil = Stencil.resolve(jnp.array([[0, 1, 2, 3]]), jnp.ones((1, 4)), positions)

        dropped = stencil.reindexed(stencil.indices, stencil.weights.at[..., -1].set(0.0))
        assert_allclose(np.asarray(dropped.weights), [[1 / 3, 1 / 3, 1 / 3, 0.0]])
        assert_array_equal(np.asarray(dropped.positions.phi), np.asarray(positions.phi))

    def test_unpositioned_has_no_sky_positions(self):
        """A grid that is not the sphere reports no positions rather than plausible wrong ones."""
        stencil = Stencil.unpositioned(
            jnp.array([[0, -1, 2, 3]]), jnp.array([[0.4, 0.4, 0.1, 0.1]])
        )

        assert stencil.positions is None
        # still resolved, like any other stencil
        assert_array_equal(np.asarray(stencil.indices), [[0, 0, 2, 3]])
        assert_allclose(np.asarray(stencil.weights), [[2 / 3, 0.0, 1 / 6, 1 / 6]])
        # the missing positions flatten away, so it stays a pytree jax can carry
        assert len(jax.tree.flatten(stencil)[0]) == 2

    @pytest.mark.parametrize('n_neighbors', [1, 4])
    def test_is_a_pytree_jax_can_trace_through(self, n_neighbors):
        """The samplers carry stencils through jit and scan, so it must flatten as a pytree."""
        positions = _positions((5, n_neighbors))
        stencil = Stencil.resolve(
            jnp.zeros((5, n_neighbors), jnp.int32), jnp.ones((5, n_neighbors)), positions
        )

        leaves, treedef = jax.tree.flatten(stencil)
        assert len(leaves) == 5
        assert jax.jit(lambda s: s.weights.sum())(stencil) == pytest.approx(5.0)
        assert isinstance(jax.tree.unflatten(treedef, leaves), Stencil)


class TestIntegrated:
    def _stacked(self, *parts: Stencil) -> Stencil:
        """Stack one-direction stencils along a new second-to-last axis, as a sampler builds them."""
        return Stencil(
            jnp.stack([p.indices for p in parts], axis=-2),
            jnp.stack([p.weights for p in parts], axis=-2),
            None
            if parts[0].positions is None
            else SkyPositions(
                *(
                    jnp.stack([getattr(p.positions, c) for p in parts], axis=-2)
                    for c in ['z', 'sth', 'phi']
                )
            ),
        )

    def test_folds_the_directions_and_scales_each_by_its_weight(self):
        parts = [
            Stencil.resolve(jnp.array([[0, 1]]), jnp.array([[0.5, 0.5]]), _positions((1, 2))),
            Stencil.resolve(jnp.array([[2, 3]]), jnp.array([[0.25, 0.75]]), _positions((1, 2))),
        ]
        merged = self._stacked(*parts).integrated(jnp.array([0.4, 0.6]))

        assert merged.n_neighbors == 4
        assert_array_equal(np.asarray(merged.indices), [[0, 1, 2, 3]])
        assert_allclose(np.asarray(merged.weights), [[0.2, 0.2, 0.15, 0.45]])
        for component in SkyPositions._fields:
            assert_array_equal(
                np.asarray(getattr(merged.positions, component)),
                np.concatenate(
                    [np.asarray(getattr(p.positions, component)) for p in parts], axis=-1
                ),
            )

    def test_one_direction_with_unit_weight_is_that_direction(self):
        part = Stencil.resolve(
            jnp.array([[0, -1, 2, 3]]), jnp.array([[0.4, 0.4, 0.1, 0.1]]), _positions((1, 4))
        )
        merged = self._stacked(part).integrated(jnp.array([1.0]))
        assert_array_equal(np.asarray(merged.indices), np.asarray(part.indices))
        assert_allclose(np.asarray(merged.weights), np.asarray(part.weights))

    def test_a_direction_off_the_map_renormalizes_to_the_directions_in_view(self):
        """The same convention as a partly covered bilinear sample: no bias, less support."""
        in_view = Stencil.unpositioned(jnp.array([[0, 1]]), jnp.array([[0.5, 0.5]]))
        off_map = Stencil.unpositioned(jnp.array([[-1, -1]]), jnp.array([[0.5, 0.5]]))
        merged = self._stacked(in_view, off_map).integrated(jnp.array([0.5, 0.5]))
        assert_allclose(np.asarray(merged.weights), [[0.5, 0.5, 0.0, 0.0]])
        assert merged.positions is None

    def test_per_stokes_weights_give_one_weight_row_per_component(self):
        """Each component reads the same pixels with its own weights, each row summing to one."""
        parts = [
            Stencil.unpositioned(jnp.array([[0, 1]]), jnp.array([[0.5, 0.5]])),
            Stencil.unpositioned(jnp.array([[2, 3]]), jnp.array([[0.5, 0.5]])),
        ]
        outer = jnp.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])  # I, Q, U
        merged = self._stacked(*parts).integrated(outer)

        assert merged.indices.shape == (1, 4)
        assert merged.weights.shape == (3, 1, 4)
        assert_allclose(
            np.asarray(merged.weights[:, 0]),
            [[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]],
        )

    def test_per_stokes_weights_broadcast_against_a_gathered_map(self):
        """The sampler contracts `values[:, ..., neighbours] * weights` over the trailing axis."""
        parts = [
            Stencil.unpositioned(jnp.array([[0, 1]]), jnp.array([[0.5, 0.5]])),
            Stencil.unpositioned(jnp.array([[2, 3]]), jnp.array([[0.5, 0.5]])),
        ]
        merged = self._stacked(*parts).integrated(jnp.array([[1.0, 0.0], [0.0, 1.0]]))
        sky = jnp.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
        sampled = jnp.sum(sky[:, merged.indices] * merged.weights, axis=-1)
        assert_allclose(np.asarray(sampled), [[1.5], [35.0]])

    def test_is_a_pytree_jax_can_trace_through(self):
        stencil = Stencil.resolve(
            jnp.zeros((5, 3, 4), jnp.int32), jnp.ones((5, 3, 4)), _positions((5, 3, 4))
        )
        merged = jax.jit(lambda s, w: s.integrated(w))(stencil, jnp.array([0.2, 0.3, 0.5]))
        assert merged.n_neighbors == 12
        assert merged.indices.shape == (5, 12)
        assert_allclose(np.asarray(merged.weights.sum(axis=-1)), 1.0)

    @pytest.mark.parametrize(
        'weights, match',
        [
            (jnp.array([1.0]), '2 directions'),
            (jnp.ones((3, 1, 2)), 'must have shape'),
            (jnp.array(1.0), 'must have shape'),
        ],
    )
    def test_rejects_weights_that_do_not_match_the_directions(self, weights, match):
        stencil = Stencil.unpositioned(jnp.zeros((1, 2, 2), jnp.int32), jnp.ones((1, 2, 2)))
        with pytest.raises(ValueError, match=match):
            stencil.integrated(weights)
