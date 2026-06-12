# ruff: noqa: F821
import jax.numpy as jnp
from jax import Array

from furax.mapmaking.spline_hwpss_template import (
    cubic_bspline,
    spline_4f_hwpss_basis,
    spline_4f_template,
    spline_basis,
)


def test_cubic_bspline_support() -> None:
    u: Array = jnp.linspace(-1, 5, 100)

    y: Array = cubic_bspline(u)

    # support is [0,4)
    assert jnp.allclose(y[u < 0], 0.0)
    assert jnp.allclose(y[u >= 4], 0.0)


def test_spline_basis_shape() -> None:
    t: Array = jnp.linspace(0, 10, 200)

    B: Array = spline_basis(t, n_knots=5)

    # K = n_knots + 2
    assert B.shape[0] == 7
    assert B.shape[1] == t.size


def test_4f_basis_structure() -> None:
    t = jnp.linspace(0, 10, 100)
    hwp = jnp.linspace(0, 2 * jnp.pi, 100)

    B = spline_4f_hwpss_basis(t, hwp, n_knots=3)

    K = 3 + 2

    # 2K because cos and sin blocks
    assert B.shape[0] == 2 * K
    assert B.shape[1] == 100


def test_4f_modulation_nonzero() -> None:
    t = jnp.linspace(0, 10, 100)
    hwp = jnp.linspace(0, 2 * jnp.pi, 100)

    B = spline_4f_hwpss_basis(t, hwp, n_knots=3)

    cos_part = B[0]
    sin_part = B[1]

    # should not be identical
    assert not jnp.allclose(cos_part, sin_part)


def test_template_build() -> None:
    t = jnp.linspace(0, 10, 100)
    hwp = jnp.linspace(0, 2 * jnp.pi, 100)

    tpl = spline_4f_template(
        t,
        hwp,
        n_dets=2,
        n_knots=3,
    )

    x = jnp.ones(tpl.in_structure.shape)

    y = tpl.mv(x)

    assert y.shape == (2, 100)


def test_linearity() -> None:
    t = jnp.linspace(0, 10, 100)
    hwp = jnp.linspace(0, 2 * jnp.pi, 100)

    tpl = spline_4f_template(t, hwp, n_dets=1, n_knots=3)

    x1 = jnp.ones(tpl.in_structure.shape)
    x2 = 2.0 * jnp.ones(tpl.in_structure.shape)

    y = tpl.mv(x1 + x2)
    y2 = tpl.mv(x1) + tpl.mv(x2)

    assert jnp.allclose(y, y2)
