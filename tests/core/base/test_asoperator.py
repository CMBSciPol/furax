from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from furax import asoperator
from furax.core._base import _check_params


@pytest.mark.parametrize(
    'func',
    [
        lambda x: None,
        lambda x, y=1: None,
        lambda x, /, y=1: None,
        lambda x=1, y=2: None,
        lambda x=1, *, y=2: None,
    ],
)
def test_check_params_valid(func: Callable[..., Any]) -> None:
    _check_params(func)


@pytest.mark.parametrize(
    'func',
    [
        lambda: None,
        lambda *, y: None,
        lambda *, y=1: None,
    ],
)
def test_check_params_invalid1(func: Callable[..., Any]) -> None:
    with pytest.raises(TypeError, match='at least one positional argument'):
        _check_params(func)


@pytest.mark.parametrize(
    'func',
    [
        lambda x, *, y: None,
        lambda x, /, *, y: None,
    ],
)
def test_check_params_invalid2(func: Callable[..., Any]) -> None:
    with pytest.raises(TypeError, match='keyword-only arguments without default values'):
        _check_params(func)


@pytest.mark.parametrize(
    'func',
    [
        lambda x, y: None,
        lambda x, /, y: None,
    ],
)
def test_check_params_invalid3(func: Callable[..., Any]) -> None:
    with pytest.raises(TypeError, match='only have one positional argument without default value'):
        _check_params(func)


def test_asoperator(capsys: pytest.CaptureFixture[str]) -> None:
    def func(x, y):
        print('tracing func')
        return x + y

    op = asoperator(func, in_structure=jax.ShapeDtypeStruct((), jnp.int64), y=2)
    assert op(1) == 3

    _ = op(2)
    captured = capsys.readouterr()
    assert captured.out == 'tracing func\n'


def test_asoperator_jit(capsys: pytest.CaptureFixture[str]) -> None:
    def func(x, y):
        print('tracing func')
        return x + y

    @jax.jit
    def outer(x, y):
        op = asoperator(func, in_structure=jax.ShapeDtypeStruct((), jnp.int64), y=y)
        return op(x)

    assert outer(1, 2) == 3

    _ = outer(10, 3)
    captured = capsys.readouterr()
    assert captured.out == 'tracing func\n'


def test_asoperator_transpose(capsys: pytest.CaptureFixture[str]) -> None:
    def func(x, y):
        print('tracing func')
        return x * y

    op = asoperator(func, in_structure=jax.ShapeDtypeStruct((), jnp.int64), y=2)
    assert op.T(1) == 2

    _ = op.T(2)
    captured = capsys.readouterr()
    assert captured.out == 'tracing func\n'


def test_asoperator_inverse(capsys: pytest.CaptureFixture[str]) -> None:
    def func(x, y):
        print('tracing func')
        return x * y

    op = asoperator(func, in_structure=jax.ShapeDtypeStruct((), jnp.float64), y=2)
    assert op.I(1.0) == 0.5

    _ = op.I(2.0)
    captured = capsys.readouterr()
    assert captured.out == 'tracing func\ntracing func\n'


def test_asoperator_jit_cache(capsys: pytest.CaptureFixture[str]) -> None:
    """Tests that we can pass a non-jitted function to `asoperator` and that the jitted version
    will be used for all subsequent calls to `asoperator` with the same function and structure."""

    def func(x, y):
        print('tracing func')
        return x + y

    op1 = asoperator(func, in_structure=jax.ShapeDtypeStruct((), jnp.int64), y=2)
    op2 = asoperator(func, in_structure=jax.ShapeDtypeStruct((), jnp.int64), y=3)
    _ = op1(1)
    _ = op1(2)
    _ = op2(3)
    captured = capsys.readouterr()
    assert captured.out == 'tracing func\n'
