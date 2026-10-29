"""Hashable signature that groups operators sharing a `jit` kernel."""

import jax
import jax.numpy as jnp

from furax import DiagonalOperator, HomothetyOperator, IdentityOperator
from furax.core import AdditionOperator, CompositionOperator


def test_signature_is_hashable() -> None:
    op = DiagonalOperator(jnp.arange(5.0))
    assert isinstance(hash(op._static_signature), int)


def test_same_shape_different_values_match() -> None:
    d1 = DiagonalOperator(jnp.arange(5.0))
    d2 = DiagonalOperator(jnp.arange(5.0) + 10.0)
    assert d1._static_signature == d2._static_signature


def test_different_shape_differs() -> None:
    d1 = DiagonalOperator(jnp.arange(5.0))
    d3 = DiagonalOperator(jnp.arange(7.0))
    assert d1._static_signature != d3._static_signature


def test_different_dtype_differs() -> None:
    d1 = DiagonalOperator(jnp.arange(5.0, dtype=jnp.float32))
    d2 = DiagonalOperator(jnp.arange(5.0, dtype=jnp.float64))
    assert d1._static_signature != d2._static_signature


def test_different_class_differs() -> None:
    structure = jax.ShapeDtypeStruct((5,), jnp.float32)
    d = DiagonalOperator(jnp.ones(5, jnp.float32))
    h = HomothetyOperator(1.0, in_structure=structure)
    assert d._static_signature != h._static_signature


def test_non_array_leaves_count_by_value() -> None:
    structure = jax.ShapeDtypeStruct((5,), jnp.float32)
    h2 = HomothetyOperator(2.0, in_structure=structure)
    h2_bis = HomothetyOperator(2.0, in_structure=structure)
    h3 = HomothetyOperator(3.0, in_structure=structure)
    assert h2._static_signature == h2_bis._static_signature
    assert h2._static_signature != h3._static_signature


def test_index_leaves_by_value_arrays_by_shape() -> None:
    from furax import IndexOperator

    structure = jax.ShapeDtypeStruct((3, 8), jnp.float32)
    idx1 = jnp.array([0, 2, 2])
    idx2 = jnp.array([1, 1, 0])
    same = IndexOperator((..., idx1), in_structure=structure)
    same_bis = IndexOperator((..., idx2), in_structure=structure)
    other_leaf = IndexOperator((slice(None), idx1), in_structure=structure)
    other_shape = IndexOperator((..., idx1[:2]), in_structure=structure)
    assert same._static_signature == same_bis._static_signature
    assert same._static_signature != other_leaf._static_signature
    assert same._static_signature != other_shape._static_signature


def test_identity_is_hashable() -> None:
    structure = jax.ShapeDtypeStruct((5,), jnp.float32)
    i1 = IdentityOperator(in_structure=structure)
    i2 = IdentityOperator(in_structure=structure)
    assert i1._static_signature == i2._static_signature


def test_composition_propagates_recursively() -> None:
    structure = jax.ShapeDtypeStruct((5,), jnp.float32)
    h = HomothetyOperator(2.0, in_structure=structure)
    i = IdentityOperator(in_structure=structure)
    c1 = CompositionOperator([h, i])
    c2 = CompositionOperator([h, i])
    c3 = CompositionOperator([i, h])
    assert c1._static_signature == c2._static_signature
    assert c1._static_signature != c3._static_signature


def test_addition_propagates_recursively() -> None:
    d1 = DiagonalOperator(jnp.arange(5.0))
    d2 = DiagonalOperator(jnp.arange(5.0) + 1.0)
    d3 = DiagonalOperator(jnp.arange(7.0))

    # same operands up to values
    a_same = AdditionOperator([d1, d2])
    a_same_bis = AdditionOperator([d2, d1])
    assert a_same._static_signature == a_same_bis._static_signature

    # one operand with a different shape
    a_mixed = AdditionOperator([d1, d3])
    assert a_same._static_signature != a_mixed._static_signature


def test_addition_sequential_flag_affects_signature() -> None:
    d1 = DiagonalOperator(jnp.arange(5.0))
    d2 = DiagonalOperator(jnp.arange(5.0) + 1.0)
    fused = AdditionOperator([d1, d2], sequential=False)
    sequential = AdditionOperator([d1, d2], sequential=True)
    # `sequential` is a static field, so it does participate in the signature
    assert fused._static_signature != sequential._static_signature
