from typing import get_args

import pytest

from furax.obs.stokes import ValidStokesType


@pytest.fixture(params=get_args(ValidStokesType))
def stokes(request: pytest.FixtureRequest) -> ValidStokesType:
    """Parametrized fixture for I, QU, IQU and IQUV."""
    return request.param
