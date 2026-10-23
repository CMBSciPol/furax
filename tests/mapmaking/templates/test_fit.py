import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from numpy.testing import assert_allclose

from furax.linalg import CGSolver
from furax.mapmaking.templates import Template, TemplateOperator
from furax.mapmaking.templates.bases import OffsetBasis, TensorBasis
from furax.mapmaking.templates.fit import TemplateFit
from furax.obs import TODFormat

N_DETS = 5
N_SAMPS = 24


def _setup(stokes):
    """A drift per detector and stream, and one offset common mode shared by the detectors, with
    a gain per detector, on a TOD whose weights flag some samples."""
    tod_format = TODFormat((N_DETS, N_SAMPS), jnp.float64, stokes=stokes)
    streams = None if stokes is None else stokes.lower()
    k = jr.split(jr.key(0), 4)
    drift = TensorBasis(jr.normal(k[0], (2, N_SAMPS)))
    n_streams = tod_format.n_streams
    gains = jr.uniform(k[1], (n_streams, N_DETS, 1), minval=0.5, maxval=1.5)
    templates = {
        'drift': Template(drift, tod_format, streams=streams),
        'common': Template(
            OffsetBasis(6, N_SAMPS, jnp.float64),
            tod_format,
            streams=streams,
            per=(),
            coupling=gains if stokes else gains[0],
        ),
    }
    weights = jr.uniform(k[2], (n_streams, N_DETS, N_SAMPS), minval=0.5, maxval=2.0)
    weights = weights.at[:, :, ::5].set(0.0)
    return TemplateOperator(templates), tod_format, weights, k[3]


@pytest.mark.parametrize('stokes', [None, 'QU'])
def test_fit_matches_the_dense_normal_equations(stokes):
    operator, tod_format, weights, key = _setup(stokes)
    tod = tod_format.normal(key)
    result = TemplateFit(operator, tod_format.diagonal(weights))(tod)
    assert result.converged

    T = operator.as_matrix()
    w = np.asarray(weights).ravel()
    d = np.asarray(ravel_pytree(tod)[0])
    expected = np.linalg.solve(T.T @ (w[:, None] * T), T.T @ (w * d))
    assert_allclose(ravel_pytree(result.solution)[0], expected, rtol=1e-5, atol=1e-8)


def test_fit_recovers_the_amplitudes_of_noiseless_tods():
    operator, tod_format, weights, key = _setup('QU')
    fit = TemplateFit(operator, tod_format.diagonal(weights), solver=CGSolver(rtol=1e-10))
    # one fit applies to any TOD
    for k in jr.split(key, 2):
        amplitudes = {
            name: jr.normal(kk, s.shape)
            for kk, (name, s) in zip(jr.split(k), operator.in_structure.items(), strict=True)
        }
        result = fit(operator(amplitudes))
        for name, a in amplitudes.items():
            assert_allclose(result.solution[name], a, rtol=1e-5, atol=1e-7)
        # the fitted model is the TOD
        assert_allclose(
            tod_format.as_array(fit.model(result.solution)),
            tod_format.as_array(operator(amplitudes)),
            atol=1e-7,
        )


def test_regularization_keeps_unconstrained_amplitudes_finite():
    operator, tod_format, weights, key = _setup(None)
    weights = weights.at[:, 0].set(0.0)  # no sample constrains the first detector's drift
    tod = tod_format.normal(key)
    result = TemplateFit(operator, tod_format.diagonal(weights), regularization=1e-6)(tod)
    drift = result.solution['drift']
    assert jnp.all(jnp.isfinite(drift))
    assert_allclose(drift[0], 0.0, atol=1e-9)
