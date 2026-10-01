import numpy as np

from furax.obs.sampling import DiscretizedBeam
from furax.obs.stokes import Stokes

from .config import PointingConfig


def load_beam(config: PointingConfig) -> DiscretizedBeam | None:
    """Read the beam of a pointing configuration, or `None` when it has none.

    The checks run on host, before any trace, and reject the beams the mapmakers cannot model:

    - nodes per detector: a file holds one detector set, while observations differ in their
      detectors and their order;
    - different Q and U weights: the mapmakers apply the beam in the boresight frame, where such
      weights would not act in the detector's polarization basis.

    Args:
        config: The pointing configuration, whose `beam` is the path of the beam file.
    """
    if config.beam is None:
        return None
    beam = DiscretizedBeam.load(config.beam)
    if len(beam.nodes.shape) != 1:
        raise ValueError(
            f'beam nodes in {config.beam!r} have shape {beam.nodes.shape}: the mapmakers accept '
            'only nodes shared by every detector, of shape (n_nodes,)'
        )
    weights = beam.weights
    if (
        isinstance(weights, Stokes)
        and 'Q' in weights.stokes
        and not np.array_equal(np.asarray(weights.q), np.asarray(weights.u))
    ):
        raise ValueError(
            f'beam weights in {config.beam!r} differ between Q and U: the mapmakers accept '
            'only equal Q and U weights'
        )
    return beam
