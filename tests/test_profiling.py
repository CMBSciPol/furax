import jax
import jax.numpy as jnp
import numpy as np
import pytest

from furax import DiagonalOperator, IdentityOperator
from furax.obs.stokes import StokesIQU
from furax.profiling import (
    DeviceBalance,
    ProfileReport,
    _normalize_cost_analysis,
    _real_float_dtype,
    format_bytes,
    measure_balance,
)
from furax.tree import nbytes

_F32 = jnp.float32


def _diagonal_operator(n: int, dtype: type = _F32) -> DiagonalOperator:
    structure = jax.ShapeDtypeStruct((n,), dtype)
    return DiagonalOperator(jnp.ones(n, dtype), in_structure=structure)


@pytest.mark.parametrize('n', [16, 1024])
def test_diagonal_counts_are_exact(n: int) -> None:
    report = _diagonal_operator(n).profile(measure=False)
    assert report.flops == n  # one multiply per element
    assert report.bytes_accessed == 3 * n * 4  # input, diagonal, output
    assert report.transcendentals == 0
    assert report.cost_available


def test_operator_arrays_are_counted_as_traffic() -> None:
    """The operator's own arrays must be traced arguments, not closed-over XLA constants.

    Lowering `op.mv` instead of `lambda op, x: op.mv(x)` drops the diagonal from the byte count and
    lets constant-folding delete the multiply outright, which silently halves every intensity this
    module reports.
    """
    n = 1024
    op = _diagonal_operator(n)
    report = op.profile(measure=False)

    passthrough_bytes = nbytes(op.in_structure) + nbytes(op.out_structure)
    assert report.bytes_accessed > passthrough_bytes
    assert report.bytes_accessed == passthrough_bytes + nbytes(op.diagonal)
    assert report.argument_bytes > nbytes(op.in_structure)


def test_identity_reports_zero_flops() -> None:
    # a kernel that only moves data has no 'flops' key in the cost analysis at all
    report = IdentityOperator(in_structure=jax.ShapeDtypeStruct((256,), _F32)).profile(
        measure=False
    )
    assert report.cost_available
    assert report.flops == 0.0
    assert report.bytes_accessed > 0
    assert report.arithmetic_intensity == 0.0


@pytest.mark.parametrize(
    'structure',
    [
        pytest.param(
            {'a': jax.ShapeDtypeStruct((4,), _F32), 'b': jax.ShapeDtypeStruct((2, 3), _F32)},
            id='dict',
        ),
        pytest.param(
            [jax.ShapeDtypeStruct((4,), _F32), jax.ShapeDtypeStruct((5,), _F32)], id='list'
        ),
        pytest.param(StokesIQU.structure_for((8,), _F32), id='StokesIQU'),
    ],
)
def test_pytree_structures_need_no_special_casing(structure: object) -> None:
    report = IdentityOperator(in_structure=structure).profile(measure=False)
    assert report.bytes_accessed > 0
    assert str(report)


def test_report_without_balance_leaves_the_bound_unknown() -> None:
    report = _diagonal_operator(64).profile(measure=False)
    assert report.balance is None
    assert report.is_memory_bound is None
    assert report.attainable_flops is None
    assert report.efficiency is None
    assert 'unknown' in str(report)


def test_arithmetic_intensity_of_an_empty_computation() -> None:
    report = ProfileReport(0.0, 0.0, 0.0, 0, 0, 0, 0)
    assert report.arithmetic_intensity == 0.0


def _report(flops: float, bytes_accessed: float, balance: DeviceBalance) -> ProfileReport:
    return ProfileReport(flops, 0.0, bytes_accessed, 0, 0, 0, 0, balance=balance)


@pytest.fixture
def balance() -> DeviceBalance:
    # ridge = 1000 / 100 = 10 flop/byte
    return DeviceBalance('test-device', np.dtype(np.float32), 1000.0, 100.0)


def test_ridge_point(balance: DeviceBalance) -> None:
    assert balance.ridge == 10.0
    assert 'test-device' in str(balance)


def test_roofline_below_the_ridge(balance: DeviceBalance) -> None:
    report = _report(10.0, 10.0, balance)  # intensity 1 flop/byte
    assert report.arithmetic_intensity == 1.0
    assert report.is_memory_bound
    assert report.attainable_flops == 100.0  # bandwidth-limited: 1 flop/byte x 100 byte/s
    assert report.efficiency == 0.1
    assert 'memory-bound' in str(report)


def test_a_computation_without_arithmetic_is_not_reported_as_inefficient(
    balance: DeviceBalance,
) -> None:
    report = _report(0.0, 10.0, balance)
    assert report.is_memory_bound
    assert 'pure data movement' in str(report)
    assert '0.0% of peak' not in str(report)


def test_roofline_above_the_ridge(balance: DeviceBalance) -> None:
    report = _report(1000.0, 10.0, balance)  # intensity 100 flop/byte
    assert report.arithmetic_intensity == 100.0
    assert not report.is_memory_bound
    assert report.attainable_flops == 1000.0  # clamped at peak, not 100 x 100
    assert report.efficiency == 1.0
    assert 'compute-bound' in str(report)


@pytest.mark.parametrize(
    'cost, expected, available',
    [
        pytest.param({'flops': 2.0}, {'flops': 2.0}, True, id='dict'),
        pytest.param([{'flops': 2.0}, {'flops': 8.0}], {'flops': 2.0}, True, id='list'),
        pytest.param(None, {}, False, id='unsupported-backend'),
        pytest.param([], {}, False, id='empty-list'),
    ],
)
def test_normalize_cost_analysis(cost: object, expected: dict[str, float], available: bool) -> None:
    # the shape of `cost_analysis()` varies with the backend and the jax version
    assert _normalize_cost_analysis(cost) == (expected, available)


def test_unavailable_cost_analysis_is_flagged() -> None:
    # zeros from a backend that declines to report must not read as a free computation
    report = ProfileReport(0.0, 0.0, 0.0, 0, 0, 0, 0, cost_available=False)
    assert 'unavailable' in str(report)


@pytest.mark.parametrize(
    'n, expected',
    [
        (0, '0.00B'),
        (512, '512.00B'),
        (1536, '1.50KiB'),
        (1024**2, '1.00MiB'),
        (3 * 1024**3, '3.00GiB'),
        (1024**5, '1.00PiB'),
    ],
)
def test_format_bytes(n: int, expected: str) -> None:
    assert format_bytes(n) == expected


@pytest.mark.parametrize(
    'dtype, expected',
    [
        (np.float32, np.float32),
        (np.float64, np.float64),  # the test session enables x64
        (np.complex64, np.float32),  # measured with the component dtype
        (np.complex128, np.float64),
        (np.int32, np.float32),  # the microbenchmarks measure floating-point throughput
    ],
)
def test_real_float_dtype(dtype: type, expected: type) -> None:
    assert _real_float_dtype(dtype) == np.dtype(expected)


# the mapmaking pipeline runs with x64 disabled when `double_precision=False`, where a float64
# array cannot be allocated at all; the autouse fixture forces x64 on, hence the subprocess
@pytest.mark.insubprocess
def test_real_float_dtype_downgrades_float64_without_x64() -> None:
    jax.config.update('jax_enable_x64', False)
    assert _real_float_dtype(np.float64) == np.dtype(np.float32)
    assert _real_float_dtype(np.complex128) == np.dtype(np.float32)


@pytest.mark.slow
def test_measure_balance_is_positive_and_cached() -> None:
    balance = measure_balance(dtype=_F32)
    assert balance.peak_flops > 0
    assert balance.peak_bandwidth > 0
    assert balance.ridge > 0
    assert measure_balance(dtype=_F32) is balance  # cached, not re-measured
