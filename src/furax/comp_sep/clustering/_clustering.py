import jax_healpy as jhp
from jax import numpy as jnp
from ._kmeans import kmeans_sample
import jax
from functools import partial 

def call_back_check(n_regions, max_centroids):
    if max_centroids is not None:
        if n_regions > max_centroids:
            raise ValueError("""
            in function [get_clusters] in the comp_sep module
            Number n_regions is greater than max_centroids
            Either : 
            - Increase max_centroids
            - Set max_centroids to None but n_regions will have 
              to be static and can no longer be a tracer
            """)

@partial(jax.jit, static_argnums=(2))
def get_cutout_from_mask(ful_map, indices , axis=0):
    return jax.tree.map(lambda x: jnp.take(x, indices,axis=axis), ful_map)

@partial(jax.jit, static_argnums=(2))
def from_cutout_to_fullmap(labels, indices, nside):
    npix = 12 * nside ** 2
    ipix = jnp.arange(npix)
    map_ids = jax.tree.map(lambda x :  jnp.full(npix, jhp.UNSEEN) , labels)
    return jax.tree.map(lambda maps, lbl : maps.at[indices].set(lbl), map_ids, labels)


@partial(jax.jit, static_argnums=(4 , 5))
def get_clusters(mask, indices, n_regions, key, max_centroids=None, unassigned=jhp.UNSEEN):
    jax.debug.callback(call_back_check, n_regions, max_centroids)

    npix = mask.size
    nside = jhp.npix2nside(npix)
    ipix = jnp.arange(npix)
    ra, dec = jhp.pix2ang(nside, ipix, lonlat=True)
    goodpix = indices
    ra_dec = jnp.stack([ra[goodpix], dec[goodpix]], axis=-1)
    km = kmeans_sample(key, ra_dec, n_regions, max_centroids=max_centroids, maxiter=100, tol=1.0e-5)
    map_ids = jnp.full(npix, unassigned)
    return map_ids.at[ipix[goodpix]].set(km.labels)


