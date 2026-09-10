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
    """Every constructor resolves the stencil it builds; `Stencil.scalar` is the shortest one."""

    def test_normalizes_the_weights(self):
        stencil = Stencil.scalar(jnp.array([[0, 1, 2, 3]]), jnp.array([[1.0, 1.0, 1.0, 1.0]]))
        assert_allclose(np.asarray(stencil.weights), 0.25)

    def test_out_of_map_neighbours_are_dropped_and_the_rest_rescaled(self):
        stencil = Stencil.scalar(jnp.array([[0, -1, 2, 3]]), jnp.array([[0.4, 0.4, 0.1, 0.1]]))
        assert_array_equal(np.asarray(stencil.indices), [[0, 0, 2, 3]])
        assert_allclose(np.asarray(stencil.weights), [[2 / 3, 0.0, 1 / 6, 1 / 6]])

    def test_a_fully_uncovered_sample_is_zero_rather_than_nan(self):
        """Dividing by a zero weight sum must be guarded: such a sample contributes nothing."""
        stencil = Stencil.scalar(jnp.array([[-1, -1]]), jnp.array([[0.7, 0.3]]))
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

    def test_resolve_casts_to_the_requested_dtype(self):
        positions = _positions((3, 4))
        stencil = Stencil.resolve(
            jnp.zeros((3, 4), jnp.int32), jnp.ones((3, 4)), positions, dtype=jnp.float32
        )
        assert stencil.weights.dtype == jnp.float32
        assert all(component.dtype == jnp.float32 for component in stencil.positions)
        assert stencil.indices.dtype == jnp.int32

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

    def test_scalar_has_no_sky_positions(self):
        """A grid that is not the sphere reports no positions rather than plausible wrong ones."""
        stencil = Stencil.scalar(jnp.array([[0, -1, 2, 3]]), jnp.array([[0.4, 0.4, 0.1, 0.1]]))

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
