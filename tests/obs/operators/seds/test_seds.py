from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax.obs import CMBOperator, DustOperator, SynchrotronOperator
from furax.obs.landscapes import HealpixLandscape
from furax.obs.stokes import Stokes


@pytest.fixture(scope='module')
def fg_data() -> tuple[dict[str, np.ndarray], Stokes, jax.ShapeDtypeStruct]:
    fg_filename = Path(__file__).parent / 'data/fgbuster_data.npz'

    data = np.load(fg_filename)
    freq_maps = data['freq_maps']
    d = Stokes.from_stokes(i=freq_maps[:, 0, :], q=freq_maps[:, 1, :], u=freq_maps[:, 2, :])

    nside = 32
    stokes_type = 'IQU'
    in_structure = HealpixLandscape(nside, stokes_type).structure

    return data, d, in_structure


SED_FACTORIES = {
    'CMB': lambda nu, units, s: CMBOperator(nu, in_structure=s, units=units),
    'DUST': lambda nu, units, s: DustOperator(
        nu, in_structure=s, frequency0=150.0, units=units, temperature=20.0, beta=1.54
    ),
    'SYNC': lambda nu, units, s: SynchrotronOperator(
        nu, in_structure=s, frequency0=20.0, units=units, beta_pl=-3.0
    ),
}


@pytest.mark.parametrize('units', ['K_CMB', 'K_RJ'])
@pytest.mark.parametrize('component', SED_FACTORIES)
def test_sed_matches_fgbuster(fg_data, component, units):
    data, _, in_structure = fg_data
    op = SED_FACTORIES[component](data['frequencies'], units, in_structure)
    assert_allclose(op.sed()[:, 0], data[f'{component}_{units}'])


@pytest.mark.parametrize('component', ['DUST', 'SYNC'])
def test_sed_patch_indices(fg_data, component):
    """Each pixel takes the SED of its patch."""
    data, _, in_structure = fg_data
    nu = data['frequencies']
    n_pix = in_structure.shape[-1]
    patch_indices = jnp.arange(n_pix) % 2
    if component == 'DUST':
        op = DustOperator(
            nu,
            frequency0=150.0,
            temperature=jnp.array([20.0, 15.0]),
            temperature_patch_indices=patch_indices,
            beta=jnp.array([1.54, 1.6]),
            beta_patch_indices=patch_indices,
            in_structure=in_structure,
        )
        patch1 = DustOperator(
            nu, frequency0=150.0, temperature=15.0, beta=1.6, in_structure=in_structure
        )
    else:
        op = SynchrotronOperator(
            nu,
            frequency0=20.0,
            beta_pl=jnp.array([-3.0, -2.8]),
            beta_pl_patch_indices=patch_indices,
            in_structure=in_structure,
        )
        patch1 = SynchrotronOperator(nu, frequency0=20.0, beta_pl=-2.8, in_structure=in_structure)
    patch0 = SED_FACTORIES[component](nu, 'K_CMB', in_structure)

    assert op.sed().shape == (len(nu), n_pix)
    assert_allclose(op.sed()[:, 0::2], jnp.broadcast_to(patch0.sed(), (len(nu), n_pix // 2)))
    assert_allclose(op.sed()[:, 1::2], jnp.broadcast_to(patch1.sed(), (len(nu), n_pix // 2)))


def test_synchrotron_running(fg_data):
    data, _, in_structure = fg_data
    nu = jnp.asarray(data['frequencies'])
    op = SynchrotronOperator(
        nu,
        frequency0=20.0,
        nu_pivot=23.0,
        running=0.1,
        units='K_RJ',
        beta_pl=-3.0,
        in_structure=in_structure,
    )
    expected = (nu / 20.0) ** (-3.0 + 0.1 * jnp.log(nu / 23.0))
    assert_allclose(op.sed()[:, 0], expected)


@pytest.mark.parametrize('component', SED_FACTORIES)
def test_sed_invalid_units(fg_data, component):
    data, _, in_structure = fg_data
    with pytest.raises(ValueError, match='Unknown units: K'):
        SED_FACTORIES[component](data['frequencies'], 'K', in_structure)


def test_broadcasts_sky_map_without_frequency_axis(fg_data):
    """SED operators must broadcast a genuine (no-frequency) sky map to a multi-frequency cube.

    Regression guard: ``AbstractSEDOperator.__init__`` used to declare an ``in_structure`` with
    the frequency axis already baked in, making ``in_structure == out_structure`` and breaking
    this exact call (component-separation's real usage, e.g. via ``MixingMatrixOperator``).
    """
    data, _, in_structure = fg_data
    nu = data['frequencies']

    cmb_operator = CMBOperator(nu, in_structure=in_structure, units='K_CMB')

    # The declared in_structure must stay the plain sky map (no frequency axis).
    assert cmb_operator.in_structure == in_structure
    assert cmb_operator.out_structure.shape == (len(nu),) + in_structure.shape

    x = Stokes.from_stokes(
        i=jnp.arange(in_structure.shape[0], dtype=jnp.float64),
        q=jnp.ones(in_structure.shape),
        u=-jnp.ones(in_structure.shape),
    )
    y = cmb_operator(x)

    assert y.shape == cmb_operator.out_structure.shape
    expected = cmb_operator.sed() * x.data[:, jnp.newaxis, :]
    assert jnp.allclose(y.data, expected)

    # Adjoint identity <Ax, z> == <x, A^T z>, guarding the custom mv() against a broken transpose.
    z = Stokes.from_stokes(
        *(jax.random.normal(jax.random.key(0), cmb_operator.out_structure.shape) for _ in range(3))
    )
    lhs = jnp.sum(y.data * z.data)
    rhs = jnp.sum(x.data * cmb_operator.T(z).data)
    assert jnp.allclose(lhs, rhs)
