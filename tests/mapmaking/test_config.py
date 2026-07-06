import inspect

import pytest
import yaml
from apischema import deserialize

from furax.mapmaking import config as config_module
from furax.mapmaking.config import (
    GroundConfig,
    HealpixConfig,
    LandscapeConfig,
    MapMakingConfig,
    PolynomialConfig,
    PolynomialOrders,
    SotodlibConfig,
    T2PConfig,
    TemplatesConfig,
)

# Every config class whose docstring carries an `Examples:` block.
EXAMPLE_CLASSES = [
    cls
    for _, cls in inspect.getmembers(config_module, inspect.isclass)
    if cls.__module__ == config_module.__name__ and 'Examples:' in (inspect.getdoc(cls) or '')
]


def _extract_yaml_blocks(cls: type) -> list[str]:
    """Pull every indented YAML snippet out of a class's Google-style `Examples:` section.

    Docstring layout (after `inspect.getdoc` dedents it): the `Examples:` header sits at
    column 0, each example's one-line description is indented 4 spaces, and the YAML snippet
    itself is indented 8 spaces. Blank lines only separate examples, never occur inside a
    snippet here, so any non-blank line indented less than 8 spaces closes the current block.
    """
    doc = inspect.getdoc(cls)
    assert doc is not None and 'Examples:' in doc, f'{cls.__name__} has no Examples section'
    after = doc.split('Examples:', 1)[1]

    blocks = []
    current: list[str] = []
    for line in after.split('\n'):
        if line.strip() == '':
            continue
        if line.startswith(' ' * 8):
            current.append(line[8:])
        elif current:
            blocks.append('\n'.join(current))
            current = []
    if current:
        blocks.append('\n'.join(current))
    return blocks


EXAMPLE_CASES = [
    pytest.param(cls, block, id=f'{cls.__name__}[{i}]')
    for cls in EXAMPLE_CLASSES
    for i, block in enumerate(_extract_yaml_blocks(cls))
]


@pytest.mark.parametrize('cls,yaml_block', EXAMPLE_CASES)
def test_docstring_example_parses_and_deserializes(cls: type, yaml_block: str):
    parsed = yaml.safe_load(yaml_block)
    assert isinstance(parsed, dict) and len(parsed) == 1, (
        f'expected a single top-level key, got: {parsed!r}'
    )
    (value,) = parsed.values()
    deserialize(cls, value)


class TestExplicitOnlyTemplates:
    """T2P and ground templates don't support implicit deprojection."""

    @pytest.mark.parametrize('cls', [T2PConfig, GroundConfig])
    def test_raises_if_not_explicit(self, cls):
        with pytest.raises(ValueError, match='requires explicit=True'):
            cls(explicit=False)

    @pytest.mark.parametrize('cls', [T2PConfig, GroundConfig])
    def test_accepts_explicit(self, cls):
        cls(explicit=True)


class TestT2PRequiresDemodulated:
    def test_raises_without_demodulated(self):
        with pytest.raises(ValueError, match='T2P template requires demodulated=True'):
            MapMakingConfig(templates=TemplatesConfig(t2p=T2PConfig()))

    def test_raises_without_i_leg(self):
        with pytest.raises(ValueError, match="T2P template requires an 'I' leg"):
            MapMakingConfig(
                sotodlib=SotodlibConfig(demodulated=True),
                landscape=LandscapeConfig(stokes='QU', healpix=HealpixConfig()),
                templates=TemplatesConfig(t2p=T2PConfig()),
            )

    def test_accepts_demodulated_with_i_leg(self):
        MapMakingConfig(
            sotodlib=SotodlibConfig(demodulated=True),
            landscape=LandscapeConfig(stokes='IQU', healpix=HealpixConfig()),
            templates=TemplatesConfig(t2p=T2PConfig()),
        )


class TestPolynomialLegendreQURequiresDemodulated:
    def test_raises_without_demodulated(self):
        poly = PolynomialConfig(legendre_qu=PolynomialOrders(0, 2))
        with pytest.raises(ValueError, match='legendre_qu requires demodulated=True'):
            MapMakingConfig(templates=TemplatesConfig(polynomial=poly))

    def test_accepts_demodulated(self):
        poly = PolynomialConfig(legendre_qu=PolynomialOrders(0, 2))
        MapMakingConfig(
            sotodlib=SotodlibConfig(demodulated=True),
            templates=TemplatesConfig(polynomial=poly),
        )
