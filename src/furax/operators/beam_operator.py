import jax
import jax.numpy as jnp
from furax._base.core import *
import equinox
import jax_healpy as jhp
from jaxtyping import Array, Float, PyTree, Inexact
from functools import partial
import numpy as np

@symmetric
class BeamOperator(AbstractLinearOperator):
    """
    BeamOperator applies a Gaussian beam to a map in spherical harmonic space.

    Attributes:
        fwhm (float): Full width at half maximum of the beam in radians.
        lmax (int): Maximum multipole moment.
        _in_structure (PyTree[jax.ShapeDtypeStruct]): Input structure of the operator.
        _beam_fl (Float[Array, 'a']): Beam transfer function.
        _inv_flag (bool): Flag to indicate if the inverse beam should be used.
    """
    fwhm: float = equinox.field(static=True)
    lmax: int = equinox.field(static=True)
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    _beam_fl: Float[Array, ' a'] = equinox.field(static=True)        
    _inv_flag: bool = equinox.field(static=True)

    def __init__(
        self, 
        fwhm: float, # [rad]
        in_structure: PyTree[jax.ShapeDtypeStruct],
        lmax: int = 512,
        inv_flag: bool = False,
    ) -> None:
        self.fwhm = fwhm
        self.lmax = lmax    
        self._in_structure = in_structure
        self._inv_flag = inv_flag

        self._beam_fl = self._gauss_beam(self.fwhm, self.lmax) # to be replaced with jhp.sphtfunc.gauss_beam when it exists
        if self._inv_flag:
            self._beam_fl = 1/self._beam_fl


    def _gauss_beam(self, fwhm: float, lmax: int = 512, pol: bool = False):
        # just copying code from "https://github.com/healpy/healpy/blob/dd0506a4b51c2961bea50cd9e0361db0c634dfdb/lib/healpy/sphtfunc.py#L1235"
        
        sigma = fwhm / np.sqrt(8.0 * np.log(2.0))
        ell = np.arange(lmax + 1)
        sigma2 = sigma ** 2
        g = np.exp(-0.5 * ell * (ell + 1) * sigma2)
        
        if not pol:  # temperature-only beam
            return g
        else:  # polarization beam
            # polarization factors [1, 2 sigma^2, 2 sigma^2, sigma^2]
            pol_factor = np.exp([0.0, 2 * sigma2, 2 * sigma2, sigma2])
            return g[:, np.newaxis] * pol_factor


    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        map2alm = partial(jhp.map2alm, iter=0, lmax=self.lmax, pol=False)
        almxfl = partial(jhp.almxfl, fl=self._beam_fl, lmax=self.lmax)

        def func(value):
            nside = jhp.npix2nside(value.shape[-1])
            alm2map = partial(jhp.alm2map, nside=nside, lmax=self.lmax, pol=False)

            alm = jax.vmap(map2alm)(jnp.atleast_2d(value))
            x = jax.vmap(almxfl)(jnp.atleast_2d(alm))
            mp = jax.vmap(alm2map)(jnp.atleast_2d(x))

            return mp.real

        return jax.tree.map(func, x)


    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
            return self._in_structure


    def inverse(self) -> AbstractLinearOperator:
        return BeamOperator(fwhm=self.fwhm, in_structure=self._in_structure, lmax=self.lmax, inv_flag=True)
