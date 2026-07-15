from typing import get_args

import pytest

from furax.obs.stokes import ValidStokesLiteral


@pytest.fixture(params=get_args(ValidStokesLiteral))
def stokes(request: pytest.FixtureRequest) -> ValidStokesLiteral:
    """Parametrized fixture for I, QU, IQU and IQUV."""
    return request.param
