import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax.obs.stencil import Stencil, StencilOrder, resolve_stencil


def _positions(shape: tuple[int, ...]) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Arbitrary but valid (z, sth, phi) for a stencil of the given shape."""
    rng = np.random.default_rng(0)
    theta = jnp.asarray(rng.uniform(0.1, np.pi - 0.1, shape))
    phi = jnp.asarray(rng.uniform(0.0, 2 * np.pi, shape))
    return jnp.cos(theta), jnp.sin(theta), phi


class TestStencilOrder:
    @pytest.mark.parametrize('order', list(StencilOrder))
    def test_the_value_is_the_neighbour_count(self, order):
        """The order is how many pixels the sample reads, which a stencil can be checked against."""
        z, sth, phi = _positions((3, order))
        stencil = Stencil.resolve(
            jnp.zeros((3, order), jnp.int32), jnp.ones((3, order)), z, sth, phi
        )
        assert stencil.n_neighbors == order


class TestResolveStencil:
    def test_normalizes_the_weights(self):
        indices = jnp.array([[0, 1, 2, 3]])
        indices, weights = resolve_stencil(indices, jnp.array([[1.0, 1.0, 1.0, 1.0]]))
        assert_allclose(np.asarray(weights), 0.25)

    def test_out_of_map_neighbours_are_dropped_and_the_rest_rescaled(self):
        indices, weights = resolve_stencil(
            jnp.array([[0, -1, 2, 3]]), jnp.array([[0.4, 0.4, 0.1, 0.1]])
        )
        assert_array_equal(np.asarray(indices), [[0, 0, 2, 3]])
        assert_allclose(np.asarray(weights), [[2 / 3, 0.0, 1 / 6, 1 / 6]])

    def test_a_fully_uncovered_sample_is_zero_rather_than_nan(self):
        """Dividing by a zero weight sum must be guarded: such a sample contributes nothing."""
        indices, weights = resolve_stencil(jnp.array([[-1, -1]]), jnp.array([[0.7, 0.3]]))
        assert_array_equal(np.asarray(indices), [[0, 0]])
        assert_array_equal(np.asarray(weights), [[0.0, 0.0]])


class TestStencil:
    def test_resolve_is_the_resolved_stencil(self):
        indices, weights = jnp.array([[0, -1, 2, 3]]), jnp.array([[0.4, 0.4, 0.1, 0.1]])
        z, sth, phi = _positions((1, 4))
        stencil = Stencil.resolve(indices, weights, z, sth, phi)

        ref_indices, ref_weights = resolve_stencil(indices, weights)
        assert_array_equal(np.asarray(stencil.indices), np.asarray(ref_indices))
        assert_array_equal(np.asarray(stencil.weights), np.asarray(ref_weights))
        # a re-numbering never moves a pixel, so the positions are passed through untouched
        assert_array_equal(np.asarray(stencil.z), np.asarray(z))
        assert stencil.n_neighbors == 4

    def test_resolve_casts_to_the_requested_dtype(self):
        z, sth, phi = _positions((3, 4))
        stencil = Stencil.resolve(
            jnp.zeros((3, 4), jnp.int32), jnp.ones((3, 4)), z, sth, phi, dtype=jnp.float32
        )
        assert stencil.weights.dtype == jnp.float32
        assert stencil.z.dtype == stencil.sth.dtype == stencil.phi.dtype == jnp.float32
        assert stencil.indices.dtype == jnp.int32

    def test_nearest_holds_one_neighbour_of_unit_weight(self):
        theta, phi = jnp.array([0.3, 1.2]), jnp.array([0.0, 4.0])
        stencil = Stencil.nearest(jnp.array([7, 9]), theta, phi)

        assert stencil.n_neighbors == 1
        assert stencil.indices.shape == (2, 1)
        assert_array_equal(np.asarray(stencil.indices[..., 0]), [7, 9])
        assert_array_equal(np.asarray(stencil.weights), 1.0)
        assert_allclose(np.asarray(stencil.z[..., 0]), np.cos(np.asarray(theta)))
        assert_allclose(np.asarray(stencil.sth[..., 0]), np.sin(np.asarray(theta)))
        assert_array_equal(np.asarray(stencil.phi[..., 0]), np.asarray(phi))

    def test_nearest_outside_the_map_contributes_nothing(self):
        stencil = Stencil.nearest(jnp.array([-1]), jnp.array([0.3]), jnp.array([0.0]))
        assert_array_equal(np.asarray(stencil.indices), [[0]])
        assert_array_equal(np.asarray(stencil.weights), [[0.0]])

    def test_reindexed_renormalizes_against_the_new_numbering(self):
        """Dropping a neighbour leaves the rest no longer summing to one."""
        z, sth, phi = _positions((1, 4))
        stencil = Stencil.resolve(jnp.array([[0, 1, 2, 3]]), jnp.ones((1, 4)), z, sth, phi)

        dropped = stencil.reindexed(stencil.indices, stencil.weights.at[..., -1].set(0.0))
        assert_allclose(np.asarray(dropped.weights), [[1 / 3, 1 / 3, 1 / 3, 0.0]])
        assert_array_equal(np.asarray(dropped.phi), np.asarray(phi))

    @pytest.mark.parametrize('n_neighbors', [1, 4])
    def test_is_a_pytree_jax_can_trace_through(self, n_neighbors):
        """The samplers carry stencils through jit and scan, so it must flatten as a pytree."""
        z, sth, phi = _positions((5, n_neighbors))
        stencil = Stencil.resolve(
            jnp.zeros((5, n_neighbors), jnp.int32), jnp.ones((5, n_neighbors)), z, sth, phi
        )

        leaves, treedef = jax.tree.flatten(stencil)
        assert len(leaves) == 5
        assert jax.jit(lambda s: s.weights.sum())(stencil) == pytest.approx(5.0)
        assert isinstance(jax.tree.unflatten(treedef, leaves), Stencil)
