from functools import partial
from typing import TypeAlias

import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import Array, Float

__all__ = [
    'qmul',
    'qrot',
    'get_local_meridian_angle',
]

Quat: TypeAlias = Float[Array, '4']
Vec3: TypeAlias = Float[Array, '3']


@jit
@partial(jnp.vectorize, signature='(4),(4)->(4)')
def qmul(q1: Quat, q2: Quat) -> Quat:
    """Compute quaternion multiplication."""
    r0 = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    r1 = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    r2 = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[1] + q1[3] * q2[1]
    r3 = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[2] + q1[3] * q2[0]
    return jnp.array([r0, r1, r2, r3])


@jit
@partial(vmap, in_axes=(0, None), out_axes=0)
def qrot(q: Quat, vec: Vec3) -> Vec3:
    """Rotate vector by quaternion."""
    # normalize quaternion
    q /= jnp.linalg.norm(q, axis=0)

    # https://fr.wikipedia.org/wiki/Quaternions_et_rotation_dans_l%27espace#M%C3%A9thodes_utilis%C3%A9es
    t2 = q[0] * q[1]
    t3 = q[0] * q[2]
    t4 = q[0] * q[3]
    t5 = -q[1] * q[1]
    t6 = q[1] * q[2]
    t7 = q[1] * q[3]
    t8 = -q[2] * q[2]
    t9 = q[2] * q[3]
    t10 = -q[3] * q[3]
    mat = 2 * jnp.array(
        [
            [t8 + t10, t6 - t4, t3 + t7],
            [t4 + t6, t5 + t10, t9 - t2],
            [t7 - t3, t2 + t9, t5 + t8],
        ]
    )
    return vec + mat @ vec


@jit
@vmap
def get_local_meridian_angle(q: Quat) -> Float[Array, '...']:
    """
    Compute angle between local meridian and orientation vector from quaternions.

    Assumes that the quaternions encode the rotation between the celestial frame
    and some other frame (e.g. detector or boresight frame). The "orientation vector"
    is the unit vector of the latter frame obtained by rotating the X axis of the
    celestial frame. For a detector this will be the polarization sensitive direction.
    The local meridian vector is obtained by projecting the -Z axis of the celestial
    frame onto the plane orthogonal to the pointing direction.

    partially taken from
    https://github.com/hpc4cmb/toast/blob/toast3/src/toast/ops/stokes_weights/kernels_jax.py#L19
    """
    vd = _qrot_zaxis(q)
    vo = _qrot_xaxis(q)

    # The vector orthogonal to the line of sight that is parallel
    # to the local meridian.
    dir_ang = jnp.arctan2(vd[1], vd[0])
    dir_r = jnp.sqrt(1.0 - vd[2] * vd[2])
    vm_z = -dir_r
    vm_x = vd[2] * jnp.cos(dir_ang)
    vm_y = vd[2] * jnp.sin(dir_ang)

    # Compute the rotation angle from the meridian vector to the
    # orientation vector.  The direction vector is normal to the plane
    # containing these two vectors, so the rotation angle is:
    #
    # angle = atan2((v_m x v_o) . v_d, v_m . v_o)
    #
    alpha_y = (
        vd[0] * (vm_y * vo[2] - vm_z * vo[1])
        - vd[1] * (vm_x * vo[2] - vm_z * vo[0])
        + vd[2] * (vm_x * vo[1] - vm_y * vo[0])
    )
    alpha_x = vm_x * vo[0] + vm_y * vo[1] + vm_z * vo[2]

    return jnp.arctan2(alpha_y, alpha_x)


@jit
@vmap
def _qrot_zaxis(q: Quat) -> Vec3:
    """Rotate the Z axis [0,0,1] by a given quaternion."""
    # normalize quaternion
    q_unit = q / jnp.linalg.norm(q)

    # perform the matrix multiplication
    x, y, z, w = q_unit
    return 2 * jnp.array([y * w + x * z, y * z - x * w, 0.5 - x * x - y * y])


@jit
@vmap
def _qrot_xaxis(q: Quat) -> Vec3:
    """Rotate the X axis [1,0,0] by a given quaternion."""
    # normalize quaternion
    q_unit = q / jnp.linalg.norm(q)

    # performs the matrix multiplication
    x, y, z, w = q_unit
    return 2 * jnp.array([0.5 - y * y - z * z, z * w + x * y, x * z - y * w])
