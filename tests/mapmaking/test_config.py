import inspect

import pytest
import yaml
from typedload.exceptions import TypedloadException

from furax.mapmaking import config as config_module
from furax.mapmaking._serialization import deserialize
from furax.mapmaking.config import (
    CommonModeConfig,
    GapsConfig,
    GapTreatment,
    HealpixConfig,
    HWPSynchronousConfig,
    LandscapeConfig,
    MapMakingConfig,
    Methods,
    PolynomialOrders,
    ScanSynchronousConfig,
    SotodlibConfig,
    SplineDriftConfig,
    T2PConfig,
    TemplatesConfig,
    TwoLevelConfig,
    WCSConfig,
    WeightingConfig,
    WeightingMode,
    _per_stream_groups,
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


def test_yaml_round_trip_writes_enums_by_name():
    config = MapMakingConfig(
        landscape=LandscapeConfig(wcs=WCSConfig()),
        weighting=WeightingConfig(mode=WeightingMode.IDENTITY),
    )
    text = config._to_yaml()
    assert 'projection: CAR' in text
    assert 'mode: identity' in text
    assert MapMakingConfig.load_dict(yaml.safe_load(text)) == config


def test_two_level_is_not_a_cg_option():
    config = MapMakingConfig.load_dict({'solver': {'two_level': {'rank': 5}}})
    assert config.solver.two_level == TwoLevelConfig(rank=5)
    assert 'two_level' not in config.solver.options


@pytest.mark.parametrize('which', ['small', 'large', 'both'])
def test_two_level_which(which):
    config = MapMakingConfig.load_dict({'solver': {'two_level': {'which': which}}})
    assert config.solver.two_level == TwoLevelConfig(which=which)


def test_two_level_which_rejects_unknown_target():
    with pytest.raises(TypedloadException, match='two_level'):
        MapMakingConfig.load_dict({'solver': {'two_level': {'which': 'SA'}}})


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


@pytest.mark.parametrize(
    ('make', 'match'),
    [
        (lambda: MapMakingConfig(max_buckets=0), 'max_buckets must be >= 1'),
        (lambda: SotodlibConfig(downsample=0), 'downsample must be >= 1'),
        (lambda: MapMakingConfig(detector_batch_size=-1), 'detector_batch_size must be >= 0'),
        (lambda: T2PConfig(zernike_order=-1), 'zernike_order'),
        (lambda: T2PConfig(quadrupole_order=0), 'quadrupole_order'),
        (lambda: T2PConfig(zernike_order=2, quadrupole_order=1), 'exclusive'),
        (lambda: T2PConfig(zernike_radius=0.0), 'zernike_radius'),
        (lambda: SplineDriftConfig(samples_per_knot=0), 'samples_per_knot must be >= 1'),
        (lambda: SplineDriftConfig(samples_per_knot={'qu': -1}), 'samples_per_knot must be >= 1'),
    ],
    ids=[
        'max_buckets',
        'downsample',
        'detector_batch_size',
        'zernike_order',
        'quadrupole_order',
        'zernike-quadrupole-exclusive',
        'zernike_radius',
        'samples_per_knot',
        'samples_per_knot-per-stream',
    ],
)
def test_rejects_out_of_range_values(make, match):
    with pytest.raises(ValueError, match=match):
        make()


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


@pytest.mark.parametrize('treatment', [GapTreatment.FILL, GapTreatment.NESTED])
def test_correlated_weight_with_templates_requires_inner_mask_gaps(treatment: GapTreatment):
    with pytest.raises(ValueError, match='inner_mask'):
        MapMakingConfig(
            weighting=WeightingConfig(mode=WeightingMode.TOEPLITZ),
            gaps=GapsConfig(treatment=treatment),
            detector_batch_size=0,  # gap treatments other than the inner mask need every detector
            templates=TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2)),
        )


@pytest.mark.parametrize(
    ('stokes', 'demodulated', 'match'),
    [
        ('I', False, 'Q and U only'),
        ('IQU', False, 'Q and U only'),
        ('IQUV', False, 'Q and U only'),
        ('QU', True, 'demodulated'),
    ],
)
def test_pomme_rejects(stokes: str, demodulated: bool, match: str):
    with pytest.raises(ValueError, match=match):
        MapMakingConfig(
            method=Methods.POMME,
            landscape=LandscapeConfig(stokes=stokes, healpix=HealpixConfig(nside=16)),
            sotodlib=SotodlibConfig(demodulated=True) if demodulated else None,
        )


class TestCommonMode:
    @pytest.mark.parametrize(
        ('kwargs', 'match'),
        [
            ({'samples_per_knot': 0}, 'samples_per_knot'),
            ({'quadrupole_order': 0}, 'quadrupole_order'),
            ({'radius': 0.0}, 'radius'),
        ],
    )
    def test_rejects_invalid_values(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CommonModeConfig(**kwargs)

    def test_requires_demodulated_qu_data(self):
        templates = TemplatesConfig(common_mode=CommonModeConfig())
        with pytest.raises(ValueError, match='common-mode template requires demodulated=True'):
            MapMakingConfig(templates=templates)
        with pytest.raises(ValueError, match='common-mode template requires Q and U streams'):
            MapMakingConfig(
                sotodlib=SotodlibConfig(demodulated=True),
                landscape=LandscapeConfig(stokes='I', healpix=HealpixConfig()),
                templates=templates,
            )


class TestT2PRequiresDemodulated:
    def test_raises_without_demodulated(self):
        with pytest.raises(ValueError, match='T2P template requires demodulated=True'):
            MapMakingConfig(templates=TemplatesConfig(t2p=T2PConfig()))

    def test_raises_without_qu_streams(self):
        with pytest.raises(ValueError, match='T2P template requires Q and U streams'):
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


class TestLegendrePerLeg:
    def test_raises_without_demodulated(self):
        legendre = {'qu': PolynomialOrders(0, 1)}
        templates = TemplatesConfig(scan_synchronous=ScanSynchronousConfig(legendre=legendre))
        with pytest.raises(ValueError, match='per stream requires demodulated=True'):
            MapMakingConfig(templates=templates)

    @pytest.mark.parametrize(
        ('streams', 'match'),
        [
            ([], 'names no stream'),
            (['iv'], r"names streams \['V'\] outside landscape.stokes"),
            (['qu', 'U'], r"names streams \['U'\] more than once"),
        ],
    )
    def test_raises_on_invalid_streams(self, streams: list[str], match: str):
        legendre = {group: PolynomialOrders(0, 1) for group in streams}
        with pytest.raises(ValueError, match=match):
            _demodulated(TemplatesConfig(scan_synchronous=ScanSynchronousConfig(legendre=legendre)))

    def test_round_trips_through_yaml(self):
        legendre = {'i': PolynomialOrders(0, 9), 'qu': PolynomialOrders(2, 5)}
        config = _demodulated(
            TemplatesConfig(scan_synchronous=ScanSynchronousConfig(legendre=legendre))
        )
        assert MapMakingConfig.load_dict(yaml.safe_load(config._to_yaml())) == config


@pytest.mark.parametrize(
    ('legendre', 'expected'),
    [
        (PolynomialOrders(0, 3), {'iqu': PolynomialOrders(0, 3)}),
        (
            {'i': PolynomialOrders(0, 9), 'qu': PolynomialOrders(0, 1)},
            {'i': PolynomialOrders(0, 9), 'qu': PolynomialOrders(0, 1)},
        ),
        # streams fitted alike are grouped whatever groups they are given in
        (
            {'U': PolynomialOrders(0, 1), 'i': PolynomialOrders(0, 1), 'q': PolynomialOrders(2, 3)},
            {'iu': PolynomialOrders(0, 1), 'q': PolynomialOrders(2, 3)},
        ),
        ({'qu': PolynomialOrders(2, 5)}, {'qu': PolynomialOrders(2, 5)}),
    ],
)
def test_per_stream_groups(legendre, expected):
    assert _per_stream_groups(legendre, 'IQU') == expected


class TestSplineDrift:
    def test_round_trips_through_yaml_per_stream(self):
        spline = SplineDriftConfig(samples_per_knot={'i': 2000, 'qu': 560})
        config = _demodulated(TemplatesConfig(spline_drift=spline))
        assert MapMakingConfig.load_dict(yaml.safe_load(config._to_yaml())) == config

    def test_per_stream_needs_demodulated_data(self):
        templates = TemplatesConfig(spline_drift=SplineDriftConfig(samples_per_knot={'qu': 500}))
        with pytest.raises(ValueError, match='per stream requires demodulated=True'):
            MapMakingConfig(templates=templates)
