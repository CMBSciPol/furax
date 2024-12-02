import jax_healpy as jhp
from jax import numpy as jnp , jit
from ._kmeans import kmeans_sample


def get_clusters(mask, n_regions, key, unassigned=jhp.UNSEEN):
    npix = mask.size
    nside = jhp.npix2nside(npix)
    ipix = jnp.arange(npix)
    ra, dec = jhp.pix2ang(nside, ipix, lonlat=True)
    goodpix = mask > 0
    ra_dec = jnp.stack([ra[goodpix], dec[goodpix]], axis=-1)
    km = kmeans_sample(key, ra_dec, n_regions, maxiter=100, tol=1.0e-5)
    map_ids = jnp.full(npix, unassigned)
    return map_ids.at[ipix[goodpix]].set(km.labels)
