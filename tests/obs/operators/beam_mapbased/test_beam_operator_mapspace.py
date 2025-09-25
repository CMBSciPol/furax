from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import load_npz

from furax.obs import (
    BeamOperatorMapspace,
    ListToStokesOperator,
    ReadBeamMatrix,
    StackedBeamOperator,
    StokesToListOperator,
)
from furax.obs.stokes import Stokes, StokesIQU


def setup_beam_operator_test():
    """Set up data for testing the BeamOperatorMapspace."""
    nfreq = 2
    nside = 16
    npix = 12 * nside**2

    maps = []
    for i in range(3):
        maps.append(np.random.rand(nfreq, npix))
    d = Stokes.from_stokes(I = maps[0], Q = maps[1], U = maps[2])

    test_beam_operator = ReadBeamMatrix(
        in_structure=d.structure, 
        path_to_file= Path(__file__).parent / "data/beam_sparse_16_FWHM40.0_cutoff.npz"
    )
    assert isinstance(test_beam_operator, BeamOperatorMapspace)

    test_matrix = load_npz(Path(__file__).parent / "data/beam_sparse_16_FWHM40.0_cutoff.npz")

    return d, test_beam_operator, test_matrix

def test_BeamOperatorMapspace():
    """Test matrix-vector multiplication."""
    d, B_op, A = setup_beam_operator_test()

    map_out = B_op.mv(d.i[0])
    map_out_mat = A.dot(np.array(d.i[0]))

    assert jnp.allclose(map_out, map_out_mat)

def test_BeamOperatorMapspace_transpose():
    """Test transpose matrix-vector multiplication."""
    d, B_op, A = setup_beam_operator_test()

    map_out = B_op.T.mv(d.i[0])
    map_out_mat = A.T.dot(np.array(d.i[0]))

    assert jnp.allclose(map_out, map_out_mat, rtol=1e-2)

def test_BeamOperatorMapspace_inverse():
    """Test inverse matrix-vector multiplication."""
    d, B_op, A = setup_beam_operator_test()

    map_out = B_op.I.mv(d.i[0])
    map_out_mat = jnp.linalg.solve(A.toarray(), np.array(d.i[0]))

    assert jnp.allclose(map_out, map_out_mat, atol=1e-4)

def setup_stokes_conversion_test():
    """Setup function for Stokes conversion tests."""
    n_freq = 3
    n_pix = 5
    axis = 0
    
    # Create test StokesIQU data
    stokes_data = StokesIQU(
        i=jnp.ones((n_freq, n_pix)),
        q=jnp.ones((n_freq, n_pix)) * 2,
        u=jnp.ones((n_freq, n_pix)) * 3
    )
    
    in_structure = jax.ShapeDtypeStruct(shape=(n_freq, n_pix), dtype=jnp.float32)
    
    return {
        'n_freq': n_freq,
        'n_pix': n_pix,
        'axis': axis,
        'stokes_data': stokes_data,
        'in_structure': in_structure
    }

def test_stokes_to_list_operator_mv():
    """Test conversion from StokesIQU to list of StokesIQU."""
    setup_data = setup_stokes_conversion_test()
    
    operator = StokesToListOperator(axis=setup_data['axis'], in_structure=setup_data['in_structure'])
    result = operator.mv(setup_data['stokes_data'])
    
    assert isinstance(result, list)
    assert len(result) == setup_data['n_freq']
    
    for i, stokes in enumerate(result):
        assert stokes.i.shape == (setup_data['n_pix'],)
        assert stokes.q.shape == (setup_data['n_pix'],)
        assert stokes.u.shape == (setup_data['n_pix'],)
        
        # Check that values are correctly extracted
        np.testing.assert_array_equal(stokes.i, jnp.ones(setup_data['n_pix']))
        np.testing.assert_array_equal(stokes.q, jnp.ones(setup_data['n_pix']) * 2)
        np.testing.assert_array_equal(stokes.u, jnp.ones(setup_data['n_pix']) * 3)

def setup_list_to_stokes_test():
    """Setup function for list to Stokes conversion tests."""
    n_freq = 3
    n_pix = 5
    axis = 0
    
    # Create test list of StokesIQU data
    stokes_list = [
        StokesIQU(
            i=jnp.ones(n_pix) * (i + 1),
            q=jnp.ones(n_pix) * (i + 2),
            u=jnp.ones(n_pix) * (i + 3)
        )
        for i in range(n_freq)
    ]
    
    in_structure = jax.ShapeDtypeStruct(shape=(n_pix,), dtype=jnp.float32)
    
    return {
        'n_freq': n_freq,
        'n_pix': n_pix,
        'axis': axis,
        'stokes_list': stokes_list,
        'in_structure': in_structure
    }

def test_list_to_stokes_operator_mv():
    """Test conversion from list of StokesIQU to StokesIQU."""
    setup_data = setup_list_to_stokes_test()
    
    operator = ListToStokesOperator(axis=setup_data['axis'], in_structure=setup_data['in_structure'])
    result = operator.mv(setup_data['stokes_list'])
    
    assert result.i.shape == (setup_data['n_freq'], setup_data['n_pix'])
    assert result.q.shape == (setup_data['n_freq'], setup_data['n_pix'])
    assert result.u.shape == (setup_data['n_freq'], setup_data['n_pix'])
    
    # Check that values are correctly stacked
    for i in range(setup_data['n_freq']):
        np.testing.assert_array_equal(result.i[i], jnp.ones(setup_data['n_pix']) * (i + 1))
        np.testing.assert_array_equal(result.q[i], jnp.ones(setup_data['n_pix']) * (i + 2))
        np.testing.assert_array_equal(result.u[i], jnp.ones(setup_data['n_pix']) * (i + 3))

def test_stacked_beam_operator():
    d, B_op, A = setup_beam_operator_test()

    beam_operators = [
        StokesIQU(i = B_op, q = B_op, u = B_op),
        StokesIQU(i = B_op, q = B_op, u = B_op),
    ]

    test_StackedOperator = StackedBeamOperator(
            beam_operators=beam_operators, in_structure=d.structure)
    
    d_out = test_StackedOperator.mv(d)
    d_out_mat = StokesIQU(
        i = jnp.array([A.dot(np.array(d.i[0])), A.dot(np.array(d.i[1]))]),
        q = jnp.array([A.dot(np.array(d.q[0])), A.dot(np.array(d.q[1]))]),
        u = jnp.array([A.dot(np.array(d.u[0])), A.dot(np.array(d.u[1]))]),
    )

    assert jax.tree.all(jax.tree.map(jnp.allclose, d_out, d_out_mat))

def test_stacked_beam_operator_transpose():
    d, B_op, A = setup_beam_operator_test()

    beam_operators = [
        StokesIQU(i = B_op, q = B_op, u = B_op),
        StokesIQU(i = B_op, q = B_op, u = B_op),
    ]

    test_StackedOperator = StackedBeamOperator(
            beam_operators=beam_operators, in_structure=d.structure)
    
    d_out = test_StackedOperator.T.mv(d)
    d_out_mat = StokesIQU(
        i = jnp.array([A.T.dot(np.array(d.i[0])), A.T.dot(np.array(d.i[1]))]),
        q = jnp.array([A.T.dot(np.array(d.q[0])), A.T.dot(np.array(d.q[1]))]),
        u = jnp.array([A.T.dot(np.array(d.u[0])), A.T.dot(np.array(d.u[1]))]),
    )

    allclose_with_tol = partial(jnp.allclose, rtol=1e-2)
    assert jax.tree.all(jax.tree.map(allclose_with_tol, d_out, d_out_mat))

def test_stacked_beam_operator_inverse():
    d, B_op, A = setup_beam_operator_test()

    beam_operators = [
        StokesIQU(i = B_op, q = B_op, u = B_op),
        StokesIQU(i = B_op, q = B_op, u = B_op),
    ]

    test_StackedOperator = StackedBeamOperator(
            beam_operators=beam_operators, in_structure=d.structure)
    
    d_out = test_StackedOperator.I.mv(d)
    A_inv = np.linalg.inv(A.toarray())
    d_out_mat = StokesIQU(
        i = jnp.array([A_inv.dot(np.array(d.i[0])), A_inv.dot(np.array(d.i[1]))]),
        q = jnp.array([A_inv.dot(np.array(d.q[0])), A_inv.dot(np.array(d.q[1]))]),
        u = jnp.array([A_inv.dot(np.array(d.u[0])), A_inv.dot(np.array(d.u[1]))]),
    )

    allclose_with_tol = partial(jnp.allclose, atol=1e-4)
    assert jax.tree.all(jax.tree.map(allclose_with_tol, d_out, d_out_mat))