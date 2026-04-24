''' Function for spherical harmonic transform. '''

import jax.numpy as jnp
import jax
from jaxtyping import PyTree, Array, Inexact

import jax_healpy as jhp
from dataclasses import field

from furax import AbstractLinearOperator, orthogonal, IdentityOperator
from furax.obs.stokes import StokesIQU
from furax.core.rules import AbstractBinaryRule, NoReduction


class Map2Alm(AbstractLinearOperator):
    ''' Spherical harmonic transform. '''

    lmax: int
    nside: int

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        def func(value):
            alm = jhp.map2alm(value, iter=0, lmax=self.lmax, pol=False) # iter, pol are not implemented
            return alm
        
        return jax.tree.map(func, x)
    
    def transpose(self) -> 'Alm2Map':
        return Alm2Map(lmax=self.lmax, nside=self.nside, in_structure=self.in_structure)
    
    def inverse(self):
        # orthogonality
        return Alm2Map(lmax=self.lmax, nside=self.nside, in_structure=self.in_structure)
    

class Alm2Map(AbstractLinearOperator):
    ''' Inverse spherical harmonic transform. '''

    lmax: int
    nside: int

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        def func(value):
            map = jhp.alm2map(value, nside=self.nside, lmax=self.lmax, pol=False)
            return jnp.astype(map, jnp.float64)

        return jax.tree.map(func, x)
    
    def transpose(self) -> 'Map2Alm':
        return Map2Alm(lmax=self.lmax, nside=self.nside, in_structure=self.in_structure)
    
    def inverse(self):
        # orthogonality
        return Map2Alm(lmax=self.lmax, nside=self.nside, in_structure=self.in_structure)
    

class SHTRule(AbstractBinaryRule):
    ''' Rule for spherical harmonic transform. '''

    left_operator_class = (Map2Alm, Alm2Map)
    right_operator_class = (Alm2Map, Map2Alm)

    def apply(self, op1: AbstractLinearOperator, op2: AbstractLinearOperator) -> list[AbstractLinearOperator]:
        if isinstance(op1, Map2Alm) and isinstance(op2, Alm2Map):
            return [IdentityOperator(in_structure=op1.in_structure)]
        elif isinstance(op1, Alm2Map) and isinstance(op2, Map2Alm):
            return [IdentityOperator(in_structure=op1.in_structure)]
        else:
            raise NoReduction