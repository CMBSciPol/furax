import contextlib
import itertools

import jax
import pytest
from equinox import tree_equal
from jax import Array
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import PyTreeDef
from jaxtyping import PyTree
from numpy.testing import assert_allclose, assert_array_equal

import furax as fx
from furax.exceptions import StructureError
from furax.tree import _dense_to_tree, _get_outer_treedef, _tree_to_dense


@pytest.mark.parametrize(
    'x, expected_y',
    [
        (jnp.ones(2, dtype=jnp.float16), jnp.ones(2, dtype=jnp.float16)),
        (
            [jnp.ones(2, dtype=jnp.float16), jnp.ones((), dtype=jnp.float32)],
            [jnp.ones(2, dtype=jnp.float32), jnp.ones((), dtype=jnp.float32)],
        ),
        (jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((2,), dtype=jnp.float16)),
        (
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
            [
                jax.ShapeDtypeStruct((2,), dtype=jnp.float32),
                jax.ShapeDtypeStruct((), dtype=jnp.float32),
            ],
        ),
    ],
)
def test_as_promoted_dtype(x, expected_y) -> None:
    y = fx.tree.as_promoted_dtype(x)
    assert tree_equal(y, expected_y)


@pytest.mark.parametrize(
    'x, expected_y',
    [
        (jnp.ones(2, dtype=jnp.float16), jax.ShapeDtypeStruct((2,), dtype=jnp.float16)),
        (
            [jnp.ones(2, dtype=jnp.float16), jnp.ones((), dtype=jnp.float32)],
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
        ),
        (jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((2,), jnp.float16)),
        (
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
        ),
    ],
)
def test_as_structure(x, expected_y) -> None:
    y = fx.tree.as_structure(x)
    assert tree_equal(y, expected_y)


@pytest.mark.parametrize(
    'x, expected_y',
    [
        (jnp.ones(2, dtype=jnp.float16), jnp.zeros(2, dtype=jnp.float16)),
        (
            [jnp.ones(2, dtype=jnp.float16), jnp.ones((), dtype=jnp.float32)],
            [jnp.zeros(2, dtype=jnp.float16), jnp.zeros((), dtype=jnp.float32)],
        ),
        (jax.ShapeDtypeStruct((2,), jnp.float16), jnp.zeros(2, dtype=jnp.float16)),
        (
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
            [jnp.zeros(2, dtype=jnp.float16), jnp.zeros((), dtype=jnp.float32)],
        ),
    ],
)
def test_zeros_like(x, expected_y) -> None:
    y = fx.tree.zeros_like(x)
    assert tree_equal(y, expected_y)


@pytest.mark.parametrize(
    'x, expected_y',
    [
        (jnp.zeros(2, dtype=jnp.float16), jnp.ones(2, dtype=jnp.float16)),
        (
            [jnp.zeros(2, dtype=jnp.float16), jnp.zeros((), dtype=jnp.float32)],
            [jnp.ones(2, dtype=jnp.float16), jnp.ones((), dtype=jnp.float32)],
        ),
        (jax.ShapeDtypeStruct((2,), jnp.float16), jnp.ones(2, dtype=jnp.float16)),
        (
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
            [jnp.ones(2, dtype=jnp.float16), jnp.ones((), dtype=jnp.float32)],
        ),
    ],
)
def test_ones_like(x, expected_y) -> None:
    y = fx.tree.ones_like(x)
    assert tree_equal(y, expected_y)


@pytest.mark.parametrize(
    'x, expected_y',
    [
        (jnp.zeros(2, dtype=jnp.float16), jnp.full(2, 3, dtype=jnp.float16)),
        (
            [jnp.zeros(2, dtype=jnp.float16), jnp.zeros((), dtype=jnp.float32)],
            [jnp.full(2, 3, dtype=jnp.float16), jnp.full((), 3, dtype=jnp.float32)],
        ),
        (jax.ShapeDtypeStruct((2,), jnp.float16), jnp.full(2, 3, dtype=jnp.float16)),
        (
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
            [jnp.full(2, 3, dtype=jnp.float16), jnp.full((), 3, dtype=jnp.float32)],
        ),
    ],
)
def test_full_like(x, expected_y) -> None:
    y = fx.tree.full_like(x, 3)
    assert tree_equal(y, expected_y)


key_from_seed = jax.random.key(0)
(key0,) = jax.random.split(key_from_seed, 1)
key1, key2 = jax.random.split(key_from_seed)


@pytest.mark.parametrize(
    'x, expected_y',
    [
        (jnp.zeros(2, dtype=jnp.float16), jax.random.normal(key0, 2, dtype=jnp.float16)),
        (
            [jnp.zeros(2, dtype=jnp.float16), jnp.zeros((), dtype=jnp.float32)],
            [
                jax.random.normal(key1, 2, dtype=jnp.float16),
                jax.random.normal(key2, (), dtype=jnp.float32),
            ],
        ),
        (jax.ShapeDtypeStruct((2,), jnp.float16), jax.random.normal(key0, 2, dtype=jnp.float16)),
        (
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
            [
                jax.random.normal(key1, 2, jnp.float16),
                jax.random.normal(key2, (), dtype=jnp.float32),
            ],
        ),
    ],
)
def test_normal_like(x, expected_y) -> None:
    y = fx.tree.normal_like(x, key_from_seed)
    assert tree_equal(y, expected_y)


@pytest.mark.distributed
@pytest.mark.parametrize('mesh_context', [True, False], ids=['in-mesh', 'no-mesh'])
@pytest.mark.parametrize('call', ['eager', 'structure', 'jit'])
@pytest.mark.parametrize('axis_type', [AxisType.Explicit, AxisType.Auto], ids=['explicit', 'auto'])
@pytest.mark.parametrize('random_like', [fx.tree.normal_like, fx.tree.uniform_like])
def test_random_like_sharding_matches_full_like(
    random_like, axis_type: AxisType, call: str, mesh_context: bool
) -> None:
    if call == 'jit' and axis_type == AxisType.Explicit and not mesh_context:
        pytest.skip('jit of an explicitly sharded input requires a mesh context')
    n = jax.device_count()
    mesh = jax.make_mesh((n,), ('i',), axis_types=(axis_type,))
    x = {'a': jax.device_put(jnp.ones((n, 2)), NamedSharding(mesh, P('i'))), 'b': jnp.ones(3)}
    if call == 'structure':
        x = fx.tree.as_structure(x)

    def both(x):
        return fx.tree.full_like(x, 0), random_like(x, key_from_seed)

    with jax.set_mesh(mesh) if mesh_context else contextlib.nullcontext():
        expected, y = jax.jit(both)(x) if call == 'jit' else both(x)
    for leaf, expected_leaf in zip(jax.tree.leaves(y), jax.tree.leaves(expected)):
        assert leaf.sharding.is_equivalent_to(expected_leaf.sharding, leaf.ndim)


@pytest.mark.parametrize(
    'x, y, expected_xy',
    [
        (jnp.ones((2,)), jnp.full((2,), 3), 6),
        ({'a': -1}, {'a': 2}, -2),
        (
            {'a': jnp.ones((2,)), 'b': jnp.array([1, 0, 1])},
            {'a': jnp.full((2,), 3), 'b': jnp.array([1, 0, -1])},
            6,
        ),
    ],
)
def test_dot(x, y, expected_xy) -> None:
    assert fx.tree.dot(x, y) == expected_xy


def test_dot_invalid_pytrees() -> None:
    with pytest.raises(ValueError, match='pytree structure error'):
        _ = fx.tree.dot({'a': 1}, {'b': 2})


@pytest.mark.parametrize(
    'x, expected_norm',
    [
        (jnp.array([3.0, 4.0]), 5.0),
        ({'a': jnp.array([3.0, 0.0]), 'b': jnp.array([0.0, 4.0])}, 5.0),
        ({'a': jnp.array([1.0, 2.0, 2.0])}, 3.0),
    ],
)
def test_norm(x, expected_norm) -> None:
    assert jnp.allclose(fx.tree.norm(x), expected_norm)


@pytest.mark.parametrize(
    'structure, a, x, expected_y',
    [
        (
            jax.tree.structure({'r1': 0, 'r2': 0}),
            # a = [ 2 3 ]
            #     [ 4 5 ]
            {'r1': {'c1': 2, 'c2': 3}, 'r2': {'c1': 4, 'c2': 5}},
            {'c1': jnp.arange(3), 'c2': -1},
            {'r1': jnp.array([-3, -1, 1]), 'r2': jnp.array([-5, -1, 3])},
        ),
        (
            jax.tree.structure({'i': 0, 'q': 0, 'u': 0}),
            #     [ 1 -1 0]
            # a = [ 1  1 0]
            #     [ 0  0 1]
            {
                'i': {'i': 1, 'q': -1, 'u': 0},
                'q': {'i': 1, 'q': 1, 'u': 0},
                'u': {'i': 0, 'q': 0, 'u': 1},
            },
            {'i': 1, 'q': -1, 'u': 3},
            {'i': 2, 'q': 0, 'u': 3},
        ),
    ],
)
def test_matvec(structure, a, x, expected_y) -> None:
    actual_y = fx.tree.matvec(structure, a, x)
    assert tree_equal(actual_y, expected_y)


@pytest.mark.parametrize(
    'structure, a, x, expected_y',
    [
        (
            jax.tree.structure({'r1': 0, 'r2': 0}),
            # a = [ 2 3 ]
            #     [ 4 5 ]
            {'r1': {'c1': 2, 'c2': 3}, 'r2': {'c1': 4, 'c2': 5}},
            {'r1': jnp.arange(3), 'r2': -1},
            {'c1': jnp.array([-4, -2, 0]), 'c2': jnp.array([-5, -2, 1])},
        ),
        (
            jax.tree.structure({'i': 0, 'q': 0, 'u': 0}),
            #     [ 1 -1 0]
            # a = [ 1  1 0]
            #     [ 0  0 1]
            {
                'i': {'i': 1, 'q': -1, 'u': 0},
                'q': {'i': 1, 'q': 1, 'u': 0},
                'u': {'i': 0, 'q': 0, 'u': 1},
            },
            {'i': 1, 'q': -1, 'u': 3},
            {'i': 0, 'q': -2, 'u': 3},
        ),
    ],
)
def test_vecmat(structure, a, x, expected_y) -> None:
    actual_y = fx.tree.vecmat(x, structure, a)
    assert tree_equal(actual_y, expected_y)


@pytest.mark.parametrize(
    'a_structure, a, b_structure, b, expected_mat',
    [
        (
            jax.tree.structure({'r1': 0, 'r2': 0}),
            # a = [ 1 -1  2 ]
            #     [ 0  2 -1 ]
            {'r1': {'i': 1, 'q': -1, 'u': 2}, 'r2': {'i': 0, 'q': 2, 'u': -1}},
            jax.tree.structure({'i': 0, 'q': 0, 'u': 0}),
            #     [  1 -1 ]
            # b = [  3  0 ]
            #     [ -1  1 ]
            {'i': {'c1': 1, 'c2': -1}, 'q': {'c1': 3, 'c2': 0}, 'u': {'c1': -1, 'c2': 1}},
            {'r1': {'c1': -4, 'c2': 1}, 'r2': {'c1': 7, 'c2': -1}},
        ),
    ],
)
def test_matmat(a_structure, a, b_structure, b, expected_mat) -> None:
    actual_mat = fx.tree.matmat(a_structure, a, b_structure, b)
    assert tree_equal(actual_mat, expected_mat)


@pytest.mark.parametrize(
    'outer_structure, inner_structure',
    [
        (jax.tree.structure({'r1': 0, 'r2': 0}), jax.tree.structure({'c1': 0, 'c2': 0})),
        (jax.tree.structure([(0,)]), jax.tree.structure(([0],))),
    ],
)
def test_get_outer_treedef(outer_structure: PyTreeDef, inner_structure: PyTreeDef) -> None:
    if not isinstance(outer_structure, PyTreeDef):
        outer_structure = jax.tree.structure(outer_structure)
    if not isinstance(inner_structure, PyTreeDef):
        inner_structure_ = jax.tree.structure(inner_structure)
    else:
        inner_structure_ = inner_structure
    counter = itertools.count()
    num_outer_leaves = outer_structure.num_leaves
    outer_leaves = [
        jax.tree.unflatten(inner_structure_, inner_structure_.num_leaves * [next(counter)])
        for _ in range(num_outer_leaves)
    ]
    tree = jax.tree.unflatten(outer_structure, outer_leaves)
    assert _get_outer_treedef(inner_structure, tree) == outer_structure


@pytest.mark.parametrize(
    'outer_structure, inner_structure, tree, expected_dense',
    [
        (jax.tree.structure(0), jax.tree.structure(0), jnp.ones(10), jnp.ones((10, 1, 1))),
        (
            jax.tree.structure({'r1': 0, 'r2': 0}),
            jax.tree.structure({'c1': 0, 'c2': 0}),
            {'r1': {'c1': 1, 'c2': 2}, 'r2': {'c1': 3, 'c2': 0}},
            jnp.array([[1, 2], [3, 0]]),
        ),
        (
            jax.tree.structure({'i': 0, 'q': 0, 'u': 0}),
            jax.tree.structure({'i': 0, 'q': 0, 'u': 0}),
            {
                'i': {'i': 1, 'q': 0, 'u': 0},
                'q': {'i': 0, 'q': 2, 'u': 0},
                'u': {'i': 0, 'q': 0, 'u': jnp.array([-1, 1])},
            },
            jnp.array([[[1, 0, 0], [0, 2, 0], [0, 0, -1]], [[1, 0, 0], [0, 2, 0], [0, 0, 1]]]),
        ),
    ],
)
def test_tree_to_dense(
    outer_structure: PyTreeDef,
    inner_structure: PyTreeDef,
    tree: PyTree[Array],
    expected_dense: Array,
):
    actual_dense = _tree_to_dense(outer_structure, inner_structure, tree)
    assert_array_equal(actual_dense, expected_dense)


@pytest.mark.parametrize(
    'outer_structure, inner_structure, dense, expected_tree',
    [
        (jax.tree.structure(0), jax.tree.structure(0), jnp.ones((10, 1, 1)), jnp.ones(10)),
        (
            jax.tree.structure({'r1': 0, 'r2': 0}),
            jax.tree.structure({'c1': 0, 'c2': 0}),
            jnp.array([[1, 2], [3, 0]]),
            {
                'r1': {'c1': jnp.array(1), 'c2': jnp.array(2)},
                'r2': {'c1': jnp.array(3), 'c2': jnp.array(0)},
            },
        ),
        (
            jax.tree.structure({'i': 0, 'q': 0, 'u': 0}),
            jax.tree.structure({'i': 0, 'q': 0, 'u': 0}),
            jnp.array([[[1, 0, 0], [0, 2, 0], [0, 0, -1]], [[1, 0, 0], [0, 2, 0], [0, 0, 1]]]),
            {
                'i': {
                    'i': jnp.array([1, 1]),
                    'q': jnp.array([0, 0]),
                    'u': jnp.array([0, 0]),
                },
                'q': {
                    'i': jnp.array([0, 0]),
                    'q': jnp.array([2, 2]),
                    'u': jnp.array([0, 0]),
                },
                'u': {
                    'i': jnp.array([0, 0]),
                    'q': jnp.array([0, 0]),
                    'u': jnp.array([-1, 1]),
                },
            },
        ),
    ],
)
def test_dense_to_tree(
    outer_structure: PyTreeDef,
    inner_structure: PyTreeDef,
    dense: Array,
    expected_tree: PyTree[Array],
):
    actual_tree = _dense_to_tree(outer_structure, inner_structure, dense)
    assert tree_equal(actual_tree, expected_tree)


def _random_stacked(key: Array, k: int, dtype) -> PyTree[Array]:
    """A stacked pytree of k vectors with leaves of different shapes."""
    structure = {
        'a': jax.ShapeDtypeStruct((k, 3), dtype),
        'b': jax.ShapeDtypeStruct((k, 2, 2), dtype),
    }
    return fx.tree.normal_like(structure, key)


def _dense(X: PyTree[Array]) -> Array:
    """The (k, n) matrix whose rows are the flattened pytrees of a stacked pytree."""
    return jax.vmap(lambda x: ravel_pytree(x)[0])(X)


def test_stack_unstack_roundtrip() -> None:
    xs = [{'a': jnp.full(2, i), 'b': jnp.array(i)} for i in range(3)]
    X = fx.tree.stack(xs)
    assert tree_equal(X, {'a': jnp.array([[0, 0], [1, 1], [2, 2]]), 'b': jnp.arange(3)})
    assert tree_equal(fx.tree.unstack(X), xs)


@pytest.mark.parametrize(
    'X',
    [{'a': jnp.ones((2, 3)), 'b': jnp.ones((3, 3))}, {'a': jnp.ones((2, 3)), 'b': jnp.ones(())}],
)
def test_unstack_without_common_leading_axis(X) -> None:
    with pytest.raises(StructureError, match='leading stack axis'):
        fx.tree.unstack(X)


@pytest.mark.parametrize('as_structure', [False, True])
def test_stacked_zeros_like(as_structure: bool) -> None:
    x = {'a': jnp.ones(2, jnp.float32), 'b': jnp.ones((2, 3), jnp.complex64)}
    if as_structure:
        x = fx.tree.as_structure(x)
    X = fx.tree.stacked_zeros_like(x, 4)
    assert tree_equal(
        X, {'a': jnp.zeros((4, 2), jnp.float32), 'b': jnp.zeros((4, 2, 3), jnp.complex64)}
    )


def test_stacked_get_set() -> None:
    X = _random_stacked(jax.random.key(0), 3, jnp.float64)
    x = fx.tree.stacked_get(_random_stacked(jax.random.key(1), 1, jnp.float64), 0)

    @jax.jit
    def set_then_get(X, x, i):  # traced index
        Y = fx.tree.stacked_set(X, i, x)
        return Y, fx.tree.stacked_get(Y, i)

    Y, y = set_then_get(X, x, 1)
    assert tree_equal(y, x)
    for i in (0, 2):
        assert tree_equal(fx.tree.stacked_get(Y, i), fx.tree.stacked_get(X, i))


@pytest.mark.parametrize('index', [slice(1, 3), jnp.array([2, 0])], ids=['slice', 'index-array'])
def test_stacked_get_set_several(index) -> None:
    X = _random_stacked(jax.random.key(0), 4, jnp.float64)
    Y = _random_stacked(jax.random.key(1), 2, jnp.float64)
    assert_array_equal(_dense(fx.tree.stacked_get(X, index)), _dense(X)[index])
    Z = fx.tree.stacked_set(X, index, Y)
    assert_array_equal(_dense(Z), _dense(X).at[index].set(_dense(Y)))


def test_stacked_set_drop_out_of_bounds() -> None:
    X = _random_stacked(jax.random.key(0), 3, jnp.float64)
    x = fx.tree.stacked_get(_random_stacked(jax.random.key(1), 1, jnp.float64), 0)
    set_at = jax.jit(lambda X, x, i: fx.tree.stacked_set(X, i, x, mode='drop'))
    assert tree_equal(set_at(X, x, 3), X)
    assert tree_equal(fx.tree.stacked_get(set_at(X, x, 2), 2), x)


@pytest.mark.parametrize('dtype', [jnp.float64, jnp.complex128])
@pytest.mark.parametrize('c_shape', [(3,), (3, 2)], ids=['vector', 'matrix'])
def test_stacked_combine(dtype, c_shape) -> None:
    X = _random_stacked(jax.random.key(0), 3, dtype)
    c = jax.random.normal(jax.random.key(1), c_shape, dtype)
    Y = fx.tree.stacked_combine(X, c)
    if len(c_shape) == 1:
        assert_allclose(ravel_pytree(Y)[0], c @ _dense(X))
    else:
        assert_allclose(_dense(Y), c.T @ _dense(X))


@pytest.mark.parametrize('dtype', [jnp.float64, jnp.complex128])
def test_stacked_dot(dtype) -> None:
    X = _random_stacked(jax.random.key(0), 3, dtype)
    y = fx.tree.stacked_get(_random_stacked(jax.random.key(1), 1, dtype), 0)
    expected = _dense(X).conj() @ ravel_pytree(y)[0]
    assert_allclose(fx.tree.stacked_dot(X, y), expected)
    assert_allclose(fx.tree.stacked_dot(X, y)[1], fx.tree.dot(fx.tree.stacked_get(X, 1), y))


@pytest.mark.parametrize('dtype', [jnp.float64, jnp.complex128])
def test_stacked_gram(dtype) -> None:
    X = _random_stacked(jax.random.key(0), 3, dtype)
    Y = _random_stacked(jax.random.key(1), 2, dtype)
    assert_allclose(fx.tree.stacked_gram(X, Y), _dense(X).conj() @ _dense(Y).T)


def test_stacked_gram_projects_operator() -> None:
    X = _random_stacked(jax.random.key(0), 3, jnp.float64)
    d = fx.tree.stacked_get(_random_stacked(jax.random.key(1), 1, jnp.float64), 0)
    A = fx.BlockDiagonalOperator(
        {
            name: fx.DiagonalOperator(leaf, in_structure=fx.tree.as_structure(leaf))
            for name, leaf in d.items()
        }
    )
    projected = fx.tree.stacked_gram(X, jax.vmap(A)(X))
    assert_allclose(projected, _dense(X) @ jnp.diag(ravel_pytree(d)[0]) @ _dense(X).T)


@pytest.mark.distributed
def test_stacked_sharded() -> None:
    """Stacked pytrees sharded along the vector axis, which reductions contract over."""
    n = jax.device_count()
    mesh = jax.make_mesh((n,), ('i',), axis_types=(AxisType.Explicit,))
    X = jax.random.normal(jax.random.key(0), (3, n, 2))
    y = jax.random.normal(jax.random.key(1), (n, 2))
    c = jax.random.normal(jax.random.key(2), (3, 2))
    with jax.set_mesh(mesh):
        X_s = jax.device_put(X, NamedSharding(mesh, P(None, 'i')))
        y_s = jax.device_put(y, NamedSharding(mesh, P('i')))

        @jax.jit
        def compute(X, y):
            zeros = fx.tree.stacked_zeros_like(y, 3)
            combined = fx.tree.stacked_combine(X, c)
            return zeros, combined, fx.tree.stacked_dot(X, y), fx.tree.stacked_gram(X, X)

        zeros, combined, dots, gram = compute(X_s, y_s)

    assert zeros.sharding.spec == P(None, 'i', None), zeros.sharding.spec
    assert combined.sharding.spec == P(None, 'i', None), combined.sharding.spec
    assert_allclose(combined, jnp.einsum('kl,k...->l...', c, X), rtol=1e-12)
    assert_allclose(dots, X.reshape(3, -1) @ y.reshape(-1), rtol=1e-12)
    assert_allclose(gram, X.reshape(3, -1) @ X.reshape(3, -1).T, rtol=1e-12)
