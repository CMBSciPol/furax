# JAXIFIED Version of kmeans_radec that can be found  at https://github.com/esheldon/kmeans_radec

from functools import partial
from jax import numpy as jnp, random as jr, lax, jit
from jax.numpy import deg2rad, rad2deg, pi, sin, cos, arccos, arctan2, sqrt, newaxis
from typing import NamedTuple
import numpy as np
import jax

_TOL_DEF = 1.0e-5
_MAXITER_DEF = 100
_PIOVER2 = pi * 0.5


class KMeansState(NamedTuple):
    ra_dec: jnp.ndarray
    centroids: jnp.ndarray
    labels: jnp.ndarray
    mean_distance: jnp.ndarray
    previous_mean_distance: jnp.ndarray
    count: int


class KMeans:

    def __init__(self,
                 ncenters,
                 max_centroids=None,
                 tol=_TOL_DEF,
                 maxiter=_MAXITER_DEF):
        self.ncenters = ncenters
        self.max_centroids = max_centroids
        self.tol = tol
        self.maxiter = maxiter

    def sample_initial(self, ra_dec, key):

        if self.max_centroids is None:
            nsamples = int(
                max(2 * np.sqrt(ra_dec.shape[0]), 10 * self.ncenters))
        else:
            nsamples = int(
                max(2 * np.sqrt(ra_dec.shape[0]), 10 * self.max_centroids))
        sample_key, center_key = jr.split(key, 2)
        ra_dec_samples = random_sample(sample_key, ra_dec, nsamples)
        if self.max_centroids is None:
            centroids_samples = random_sample(center_key, ra_dec,
                                              self.ncenters)
        else:
            centroids_samples = random_sample(center_key, ra_dec,
                                              self.max_centroids)

        return ra_dec_samples, centroids_samples

    def kmeans_init(self, ra_dec, centroids):
        return KMeansState(
            ra_dec=ra_dec,
            centroids=centroids,
            labels=jnp.zeros(ra_dec.shape[0], dtype=jnp.int32),
            mean_distance=jnp.inf,
            previous_mean_distance=0.0,
            count=0,
        )

    def fit(self, ra_dec, centroids):

        centroid_mask = jnp.arange(centroids.shape[0]) < self.ncenters

        def kmeans_step(carry):
            ra_dec, indices, XYZ, state = carry
            Xs, Ys, Zs = XYZ

            # Set the previous mean distance
            state = state._replace(previous_mean_distance=state.mean_distance)

            distances = cdist_radec(ra_dec,
                                    state.centroids)  # npoints x ncenters
            if self.max_centroids is not None:
                distances = jnp.where(centroid_mask[None, :], distances,
                                      jnp.inf)

            labels = distances.argmin(axis=1).astype(
                jnp.int32)  # nearest centroid

            #jax.debug.print("labels {a}", a=labels)

            distances = distances[indices, labels]

            mean_distance = distances.mean()

            # Update the centroids
            def for_loop_body(center_indx, carry):
                centroids, labels = carry
                mask = jnp.where(labels == center_indx, 1, 0)
                # Get the 3D coordinates of the points in the cluster
                masked_X = mask * Xs
                masked_Y = mask * Ys
                masked_Z = mask * Zs
                # Compute their means in a jittable way
                mean_X = masked_X.sum() / mask.sum()
                mean_Y = masked_Y.sum() / mask.sum()
                mean_Z = masked_Z.sum() / mask.sum()

                #jax.debug.print(
                #    "At iteration {a} mask is {b} means are {c}",
                #    a=center_indx,
                #    b=mask,
                #    c=xyz2radec(mean_X, mean_Y, mean_Z))
                #
                return centroids.at[center_indx].set(
                    xyz2radec(mean_X, mean_Y, mean_Z)), labels

            new_centroids, _ = lax.fori_loop(0, self.ncenters, for_loop_body,
                                             (state.centroids, labels))

            new_state = KMeansState(
                ra_dec=ra_dec,
                centroids=new_centroids,
                labels=labels,
                mean_distance=mean_distance,
                previous_mean_distance=state.previous_mean_distance,
                count=state.count + 1,
            )

            return ra_dec, indices, XYZ, new_state

        def kmeans_continue_criteria(carry):
            _, _, _, state = carry

            # Condition for convergence
            converged = (
                (1 - self.tol) * state.previous_mean_distance
                <= state.mean_distance) & (state.previous_mean_distance
                                           >= state.mean_distance)
            #
            # Continue if not converged and within max iterations
            return (state.count < self.maxiter) & (~(converged))

        XYZ = radec2xyz(ra_dec[:, 0], ra_dec[:, 1])
        indices = jnp.arange(ra_dec.shape[0])

        init_state = self.kmeans_init(ra_dec, centroids)

        _, _, _, final_state = lax.while_loop(
            kmeans_continue_criteria, kmeans_step,
            (ra_dec, indices, XYZ, init_state))

        return final_state


def cdist_radec(a1, a2):
    """
    Compute pairwise spherical distances between two sets of points.
    """
    ra1 = a1[:, 0][:, newaxis]
    dec1 = a1[:, 1][:, newaxis]
    ra2 = a2[:, 0]
    dec2 = a2[:, 1]

    phi1, theta1 = deg2rad(ra1), _PIOVER2 - deg2rad(dec1)
    phi2, theta2 = deg2rad(ra2), _PIOVER2 - deg2rad(dec2)

    sintheta1, sintheta2 = sin(theta1), sin(theta2)
    x1, y1, z1 = sintheta1 * cos(phi1), sintheta1 * sin(phi1), cos(theta1)
    x2, y2, z2 = sintheta2 * cos(phi2), sintheta2 * sin(phi2), cos(theta2)

    costheta = x1 * x2 + y1 * y2 + z1 * z2
    costheta = jnp.clip(costheta, -1.0, 1.0)
    return arccos(costheta)


def random_sample(key, ra_dec, nsamples):
    nra_dec = ra_dec.shape[0]
    indices = jr.choice(key, nra_dec, shape=(nsamples, ), replace=False)
    return ra_dec[indices]


def get_mean_center(x, y, z):
    """
    Compute mean center from Cartesian coordinates.
    """
    xmean, ymean, zmean = x.mean(), y.mean(), z.mean()

    return xyz2radec(xmean, ymean, zmean)


def xyz2radec(x, y, z):
    """
    Convert Cartesian coordinates to spherical.
    """
    r = sqrt(x**2 + y**2 + z**2)
    theta = arccos(z / r)
    phi = arctan2(y, x)
    ra = rad2deg(phi)
    dec = rad2deg(_PIOVER2 - theta)
    ra = atbound1(ra, 0.0, 360.0)
    return jnp.array([ra, dec])


def radec2xyz(ra, dec):
    """
    Convert spherical coordinates to Cartesian.
    """
    phi, theta = deg2rad(ra), _PIOVER2 - deg2rad(dec)
    sintheta = sin(theta)
    return sintheta * cos(phi), sintheta * sin(phi), cos(theta)


def atbound1(longitude_in, minval, maxval):
    range_size = maxval - minval
    longitude = (longitude_in - minval) % range_size + minval
    return longitude


def kmeans_sample(key,
                  ra_dec,
                  ncenters,
                  max_centroids=None,
                  tol=_TOL_DEF,
                  maxiter=_MAXITER_DEF):
    km = KMeans(ncenters, max_centroids, tol, maxiter)
    ra_dec_samples, centroids_samples = km.sample_initial(ra_dec, key)

    # First run on the samples
    state = km.fit(ra_dec_samples, centroids_samples)

    # Second run on the full data
    state = km.fit(ra_dec, state.centroids)

    return state

