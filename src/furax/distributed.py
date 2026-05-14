import jax


def maybe_init() -> bool:
    try:
        jax.distributed.initialize()
        return True
    except ValueError:
        return False
