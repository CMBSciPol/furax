import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpy.testing import assert_array_equal

from furax import tree
from furax.obs import TODFormat
from furax.obs.stokes import Stokes, ValidStokesLiteral


@pytest.mark.parametrize(
    ('shape', 'n_dets'), [((3, 10), 3), ((2, 3, 10), 6), ((10,), 1)], ids=['dets', 'axes', 'single']
)
def test_shape_splits_into_detectors_and_samples(shape, n_dets):
    tod_format = TODFormat(shape, jnp.float32)
    assert tod_format.det_shape == shape[:-1]
    assert tod_format.n_dets == n_dets
    assert tod_format.n_samples == 10
    assert tod_format.structure == jax.ShapeDtypeStruct(shape, jnp.float32)


@pytest.mark.parametrize('stokes', [None, 'IQU'])
def test_streams_are_stacked_on_a_leading_axis(stokes):
    tod_format = TODFormat((2, 3, 10), jnp.float64, stokes=stokes)
    n_streams = 1 if stokes is None else 3
    assert tod_format.n_streams == n_streams
    x = tod_format.normal(jr.key(2))
    array = tod_format.as_array(x)
    assert array.shape == (n_streams, 2, 3, 10)
    assert tree.as_structure(tod_format.from_array(array)) == tod_format.structure
    assert_array_equal(jax.tree.leaves(tod_format.from_array(array)), jax.tree.leaves(x))


def test_demodulated_structure_is_a_stokes_of_streams(stokes: ValidStokesLiteral):
    tod_format = TODFormat((3, 10), jnp.float64, stokes=stokes)
    expected = Stokes.class_for(stokes).structure_for((3, 10), jnp.float64)
    assert tree.as_structure(tod_format.zeros()) == expected
    assert tod_format.structure == expected


@pytest.mark.parametrize('stokes', [None, 'IQU'])
def test_fills_and_samples_have_the_structure(stokes):
    tod_format = TODFormat((3, 10), jnp.float64, stokes=stokes)
    assert_array_equal(
        jax.tree.leaves(tod_format.full(2.5)),
        jax.tree.leaves(tree.full_like(tod_format.structure, 2.5)),
    )
    assert all(jnp.all(leaf == 0) for leaf in jax.tree.leaves(tod_format.zeros()))
    assert all(jnp.all(leaf == 1) for leaf in jax.tree.leaves(tod_format.ones()))
    for sample in (tod_format.normal(jr.key(0)), tod_format.uniform(jr.key(1), 2.0, 3.0)):
        assert tree.as_structure(sample) == tod_format.structure
    uniform = jax.tree.leaves(tod_format.uniform(jr.key(1), 2.0, 3.0))
    assert all(jnp.all((leaf >= 2) & (leaf < 3)) for leaf in uniform)


def test_equal_structures_hash_alike():
    # a static field of an operator: equal structures must compare and hash alike
    a = TODFormat([3, 10], 'float32', stokes='QU')
    b = TODFormat((3, 10), jnp.float32, stokes='QU')
    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.parametrize(
    ('kwargs', 'error'),
    [({'shape': ()}, 'at least the sample axis'), ({'stokes': 'QUV'}, 'Invalid Stokes')],
)
def test_rejects(kwargs, error):
    with pytest.raises(ValueError, match=error):
        TODFormat(**({'shape': (3, 10), 'dtype': jnp.float32} | kwargs))


@pytest.mark.parametrize('stokes', [None, 'QU'])
def test_diagonal_scales_each_stream_by_its_weights(stokes):
    tod_format = TODFormat((2, 3, 10), jnp.float64, stokes=stokes)
    weights = jr.uniform(jr.key(3), (tod_format.n_streams, 2, 3, 10))
    x = tod_format.normal(jr.key(4))
    assert_array_equal(
        tod_format.as_array(tod_format.diagonal(weights)(x)), weights * tod_format.as_array(x)
    )
