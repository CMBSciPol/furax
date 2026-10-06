import inspect

import pytest
import yaml
from typedload.exceptions import TypedloadException

from furax.mapmaking import config as config_module
from furax.mapmaking._serialization import deserialize
from furax.mapmaking.config import (
    GapsConfig,
    GapTreatment,
    GroundConfig,
    HealpixConfig,
    HWPSynchronousConfig,
    LandscapeConfig,
    MapMakingConfig,
    PolynomialConfig,
    PolynomialOrders,
    ScanSynchronousConfig,
    SotodlibConfig,
    T2PConfig,
    TemplatesConfig,
    WCSConfig,
    WeightingConfig,
    WeightingMode,
    _legendre_leg_groups,
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


def test_yaml_round_trip_writes_projection_by_name():
    config = MapMakingConfig(landscape=LandscapeConfig(wcs=WCSConfig()))
    text = config._to_yaml()
    assert 'projection: CAR' in text
    assert MapMakingConfig.load_dict(yaml.safe_load(text)) == config


def test_int_is_accepted_for_float_field():
    config = MapMakingConfig.load_dict({'solver': {'rtol': 1}})
    assert config.solver.rtol == 1.0
    assert isinstance(config.solver.rtol, float)


@pytest.mark.parametrize(
    'data',
    [
        pytest.param({'solver': {'rtl': 1e-6}}, id='unknown-key'),
        pytest.param({'solver': {'max_steps': '12'}}, id='str-for-int'),
        pytest.param({'solver': {'max_steps': 1.5}}, id='float-for-int'),
        pytest.param({'solver': {'max_steps': True}}, id='bool-for-int'),
        pytest.param({'solver': {'rtol': '1e-6'}}, id='str-for-float'),
        pytest.param({'solver': {'rtol': True}}, id='bool-for-float'),
        pytest.param({'weighting': {'mode': 'bogus'}}, id='unknown-enum-value'),
        pytest.param({'landscape': {'wcs': {'projection': 'BOGUS'}}}, id='unknown-projection'),
    ],
)
def test_load_dict_rejects_invalid_input(data: dict):
    with pytest.raises(TypedloadException):
        MapMakingConfig.load_dict(data)


@pytest.mark.parametrize(
    'templates, enabled',
    [
        (TemplatesConfig(), False),
        # a tuning knob always holds a value: setting one must not enable template fitting
        (TemplatesConfig(regularization=0.1), False),
        (TemplatesConfig(hwp_synchronous=HWPSynchronousConfig()), True),
    ],
    ids=['none-enabled', 'tuning-knob-only', 'one-enabled'],
)
def test_use_templates_follows_the_enabled_templates(templates: TemplatesConfig, enabled: bool):
    assert templates.empty == (not enabled)
    assert MapMakingConfig(templates=templates).use_templates == enabled


@pytest.mark.parametrize('max_buckets', [0, -1])
def test_max_buckets_must_be_positive(max_buckets: int):
    with pytest.raises(ValueError, match='max_buckets must be >= 1'):
        MapMakingConfig(max_buckets=max_buckets)


def test_detector_batch_size_must_not_be_negative():
    with pytest.raises(ValueError, match='detector_batch_size must be >= 0'):
        MapMakingConfig(detector_batch_size=-1)


@pytest.mark.parametrize('treatment', [GapTreatment.FILL, GapTreatment.NESTED])
def test_correlated_gap_treatments_reject_detector_batches(treatment: GapTreatment):
    correlated = WeightingConfig(mode=WeightingMode.TOEPLITZ)
    with pytest.raises(ValueError, match='set detector_batch_size to 0'):
        MapMakingConfig(weighting=correlated, gaps=GapsConfig(treatment=treatment))
    MapMakingConfig(
        weighting=correlated, gaps=GapsConfig(treatment=treatment), detector_batch_size=0
    )
    # with diagonal weights, gaps need no special treatment
    MapMakingConfig(gaps=GapsConfig(treatment=treatment), detector_batch_size=16)


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

    def test_raises_without_qu_legs(self):
        with pytest.raises(ValueError, match='T2P template requires Q and U legs'):
            MapMakingConfig(
                sotodlib=SotodlibConfig(demodulated=True),
                landscape=LandscapeConfig(stokes='I', healpix=HealpixConfig()),
                templates=TemplatesConfig(t2p=T2PConfig()),
            )

    @pytest.mark.parametrize(
        ('stokes', 't2p', 'read_stokes'),
        [('IQU', True, 'IQU'), ('QU', True, 'IQU'), ('QU', False, 'QU'), ('IQU', False, 'IQU')],
    )
    def test_reads_i_for_t2p_only(self, stokes: str, t2p: bool, read_stokes: str):
        config = MapMakingConfig(
            sotodlib=SotodlibConfig(demodulated=True),
            landscape=LandscapeConfig(stokes=stokes, healpix=HealpixConfig()),
            templates=TemplatesConfig(t2p=T2PConfig() if t2p else None),
        )
        assert config.read_stokes == read_stokes


def _demodulated(templates: TemplatesConfig) -> MapMakingConfig:
    return MapMakingConfig(
        sotodlib=SotodlibConfig(demodulated=True),
        landscape=LandscapeConfig(stokes='IQU', healpix=HealpixConfig()),
        templates=templates,
    )


@pytest.mark.parametrize('template', [PolynomialConfig, ScanSynchronousConfig])
class TestLegendrePerLeg:
    def test_raises_without_demodulated(self, template):
        name = 'polynomial' if template is PolynomialConfig else 'scan_synchronous'
        templates = TemplatesConfig(**{name: template(legendre={'qu': PolynomialOrders(0, 1)})})
        with pytest.raises(ValueError, match='per Stokes leg requires demodulated=True'):
            MapMakingConfig(templates=templates)

    @pytest.mark.parametrize(
        ('legs', 'match'),
        [
            ([], 'names no Stokes leg'),
            (['iv'], r"names legs \['V'\] outside landscape.stokes"),
            (['qu', 'U'], r"names legs \['U'\] more than once"),
        ],
    )
    def test_raises_on_invalid_legs(self, template, legs: list[str], match: str):
        name = 'polynomial' if template is PolynomialConfig else 'scan_synchronous'
        legendre = {group: PolynomialOrders(0, 1) for group in legs}
        with pytest.raises(ValueError, match=match):
            _demodulated(TemplatesConfig(**{name: template(legendre=legendre)}))

    def test_round_trips_through_yaml(self, template):
        name = 'polynomial' if template is PolynomialConfig else 'scan_synchronous'
        legendre = {'i': PolynomialOrders(0, 9), 'qu': PolynomialOrders(2, 5)}
        config = _demodulated(TemplatesConfig(**{name: template(legendre=legendre)}))
        assert MapMakingConfig.load_dict(yaml.safe_load(config._to_yaml())) == config


@pytest.mark.parametrize(
    ('legendre', 'expected'),
    [
        (PolynomialOrders(0, 3), {'iqu': PolynomialOrders(0, 3)}),
        (
            {'i': PolynomialOrders(0, 9), 'qu': PolynomialOrders(0, 1)},
            {'i': PolynomialOrders(0, 9), 'qu': PolynomialOrders(0, 1)},
        ),
        # legs fitted alike are grouped whatever groups they are given in
        (
            {'U': PolynomialOrders(0, 1), 'i': PolynomialOrders(0, 1), 'q': PolynomialOrders(2, 3)},
            {'iu': PolynomialOrders(0, 1), 'q': PolynomialOrders(2, 3)},
        ),
        ({'qu': PolynomialOrders(2, 5)}, {'qu': PolynomialOrders(2, 5)}),
    ],
)
def test_legendre_leg_groups(legendre, expected):
    assert _legendre_leg_groups(legendre, 'IQU') == expected
