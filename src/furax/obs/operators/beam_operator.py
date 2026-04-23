import jax
import jax.numpy as jnp
from furax._base.core import *
import equinox
import jax_healpy as jhp
from jaxtyping import Array, Float

class BeamOperator(AbstractLinearOperator):
    fwhm: float = equinox.field(static=True)
    lmax: int = equinox.field(static=True)
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    beam_fl: Float[Array, ' a'] = equinox.field(static=True)        
    _inv_flag: bool = equinox.field(static=True)

    def __init__(
        self, 
        fwhm: float,
        in_structure: PyTree[jax.ShapeDtypeStruct],
        lmax = 2000,
        inv_flag = False,
    ) -> None:
        self.fwhm = fwhm
        self.lmax = lmax    
        self._in_structure = in_structure
        self._inv_flag = inv_flag

        self.beam_fl = self._gauss_beam(self.fwhm, self.lmax) # to be replaced with jhp.sphtfunc.gauss_beam when it exists
        if self._inv_flag:
            self.beam_fl = 1/self.beam_fl


    def _gauss_beam(self, fwhm, lmax=512, pol=False):
        # just copying code from "https://github.com/healpy/healpy/blob/dd0506a4b51c2961bea50cd9e0361db0c634dfdb/lib/healpy/sphtfunc.py#L1235"
        
        sigma = fwhm / jnp.sqrt(8.0 * jnp.log(2.0))
        ell = jnp.arange(lmax + 1)
        sigma2 = sigma ** 2
        g = jnp.exp(-0.5 * ell * (ell + 1) * sigma2)
        
        if not pol:  # temperature-only beam
            return g
        else:  # polarization beam
            # polarization factors [1, 2 sigma^2, 2 sigma^2, sigma^2]
            pol_factor = jnp.exp([0.0, 2 * sigma2, 2 * sigma2, sigma2])
            return g[:, jnp.newaxis] * pol_factor


    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        def func(value):
            nside = jhp.npix2nside(len(value))
            value_dtype = value.dtype
            alm = jhp.map2alm(value, iter=0, lmax=self.lmax, pol=False)
            x = jhp.almxfl(alm, self.beam_fl, lmax=self.lmax)
            mp = jhp.alm2map(x, nside=nside, lmax=self.lmax, pol=False)
            return jnp.astype(mp, value_dtype)

        return jax.tree.map(func, x)


    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
            return self._in_structure


    def inverse(self) -> AbstractLinearOperator:
        return BeamOperator(fwhm=self.fwhm, in_structure=self._in_structure, lmax = self.lmax, inv_flag=True)
