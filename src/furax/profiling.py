"""Static cost estimation for jit-compiled computations.

This module provides utilities to estimate the flops, memory traffic and buffer sizes of a
computation without running it. The computation is lowered and compiled, so we can read off the
estimates from the resulting executable (see [`jax.stages.Compiled.cost_analysis`][] and
[`jax.stages.Compiled.memory_analysis`][]).

Warning:
    Those statistics are estimates computed by the underlying XLA compiler, not measurements from
    hardware counters. JAX documents it as a debugging tool whose structure "may be inconsistent
    across versions of JAX and jaxlib, or even across invocations". Read these numbers, and
    everything derived from them, as indications rather than facts.
"""

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

__all__ = [
    'Bound',
    'DeviceBalance',
    'ProfileReport',
    'device_of',
    'format_bytes',
    'measure_balance',
    'profile',
]

# Problem sizes for the microbenchmarks in `measure_balance`. Accelerators need a large matmul to
# reach peak; on CPU the same size would take seconds for no extra accuracy.
_MATMUL_SIZE = {'cpu': 1024}
_MATMUL_SIZE_DEFAULT = 4096
# element count of the arrays added together to measure bandwidth: 3 x 64 MiB of traffic in
# float32, far beyond any last-level cache, so the timing reflects main memory and not cache
_BANDWIDTH_SIZE = 1 << 24
_REPEATS = 5


def _format_quantity(n: float, unit: str, *, base: Literal[1000, 1024], sep: str = ' ') -> str:
    for prefix in ('', 'K', 'M', 'G', 'T'):
        if n < base:
            break
        n /= base
    else:
        prefix = 'P'
    if prefix and base == 1024:
        prefix += 'i'  # KiB, MiB, ... but plain B for the unprefixed case
    return f'{n:.2f}{sep}{prefix}{unit}'


def format_bytes(n: float) -> str:
    # e.g. 1536 -> '150.00KiB'
    return _format_quantity(n, 'B', base=1024, sep='')


def _format_count(n: float, unit: str) -> str:
    return _format_quantity(n, unit, base=1000)


def _real_float_dtype(dtype: DTypeLike) -> np.dtype[Any]:
    """Maps a dtype onto the real floating-point dtype whose peak rates should be measured.

    Complex dtypes are mapped to their component dtype and non-floating dtypes to float32, since
    the microbenchmarks measure floating-point throughput. float64 is downgraded when x64 is
    disabled, where float64 arrays cannot be created at all.
    """
    given = np.dtype(dtype)
    if jnp.issubdtype(given, jnp.complexfloating):
        result = np.dtype(given.type(0).real.dtype)
    elif jnp.issubdtype(given, jnp.floating):
        result = given
    else:
        result = np.dtype(np.float32)
    if result == np.float64 and not jax.config.jax_enable_x64:
        result = np.dtype(np.float32)
    return result


class Bound(StrEnum):
    """What limits a computation on a device, from [`ProfileReport.bound`][]."""

    MEMORY = 'memory'
    """The computation waits on data: its arithmetic intensity is below the ridge."""

    COMPUTE = 'compute'
    """The arithmetic sets the pace: the intensity is above the ridge."""

    AMBIGUOUS = 'ambiguous'
    """The intensity is within the ridge's measurement error, so the two cannot be told apart."""


@dataclass(frozen=True)
class DeviceBalance:
    """Peak throughputs (flops and bytes per second) of a device, for one dtype.

    A device performs only so many flops per second and moves only so many bytes per second.
    The ratio of these quantities, [`ridge`][], is how much arithmetic it must be given per byte to
    keep its arithmetic units fed (typically tens of flops per byte).

    The ridge point can be compared to a computation's [`ProfileReport.arithmetic_intensity`][],
    i.e., the flops it performs per byte it touches.

    - Below the ridge the computation is *memory-bound*: units are mostly idle, waiting for data.
    - Above the ridge it is *compute-bound*: the data arrives faster than the units can consume it.

    A matrix-vector product reads a whole matrix to do only two flops per element, so it is
    typically memory-bound on all kinds of devices; a large matrix-matrix product reuses each
    element many times, and is compute-bound on most devices.

    Attributes:
        device_kind: The `device_kind` of the [`jax.Device`][], e.g. `'NVIDIA H100 80GB HBM3'`.
        dtype: The dtype the rates were measured with.
        peak_flops: Measured peak arithmetic throughput, in flop/s.
        peak_bandwidth: Measured peak memory throughput, in byte/s.
        flops_error: Relative spread of the arithmetic measurement over its repeats.
        bandwidth_error: Relative spread of the memory measurement over its repeats.
    """

    device_kind: str
    dtype: np.dtype[Any]
    peak_flops: float
    peak_bandwidth: float
    flops_error: float = 0.0
    bandwidth_error: float = 0.0

    @property
    def ridge(self) -> float:
        """The ridge point in flop/byte: [`peak_flops`][] divided by [`peak_bandwidth`][]."""
        return self.peak_flops / self.peak_bandwidth

    @property
    def ridge_error(self) -> float:
        """Relative uncertainty on [`ridge`][]."""
        # a ratio's relative errors add in quadrature
        return float(np.hypot(self.flops_error, self.bandwidth_error))

    def __str__(self) -> str:
        return (
            f'{self.device_kind} ({self.dtype.name}) '
            f'peak={_format_count(self.peak_flops, "FLOP/s")} '
            # decimal units for consistency with flop rate
            f'bandwidth={_format_count(self.peak_bandwidth, "B/s")} '
            f'ridge={self.ridge:.3g}±{self.ridge_error:.0%} flop/byte'
        )


@dataclass(frozen=True)
class ProfileReport:
    """The estimated cost of one compiled computation.

    This is the result of a call to [`profile`][], or [`AbstractLinearOperator.profile`]
    [furax.core.AbstractLinearOperator.profile] for an operator.

    Attributes:
        flops: Floating-point operations, from XLA's cost model. Excludes `transcendentals`.
        transcendentals: Transcendental operations (`sin`, `exp`, ...), counted separately by XLA.
        bytes_accessed: Total buffer traffic, from XLA's cost model.
        argument_bytes: Total size of the executable's arguments.
        output_bytes: Total size of the executable's outputs.
        temp_bytes: Peak size of the scratch buffers, i.e. the memory needed on top of the
            arguments and outputs.
        peak_bytes: Peak device memory attributed to the executable.
        balance: A [`DeviceBalance`][] to evaluate profile counts against.
        cost_available: Whether XLA reported a cost analysis at all. When `False`, `flops`,
            `transcendentals` and `bytes_accessed` are zero because the backend declined to
            provide them, not because the computation is free.
    """

    flops: float
    transcendentals: float
    bytes_accessed: float
    argument_bytes: int
    output_bytes: int
    temp_bytes: int
    peak_bytes: int
    balance: DeviceBalance | None = None
    cost_available: bool = True

    @property
    def total_flops(self) -> float:
        """[`flops`][] plus [`transcendentals`][], counting each transcendental as one operation.

        XLA counts `sin`, `exp` and friends separately from `flops`, so a kernel that is nothing but
        transcendentals reports `flops == 0`. Charging one operation each keeps it from looking
        free, and understates it: a transcendental costs several flops. How XLA arrives at either
        counter is undocumented, so this is a convention, not a conversion.
        """
        return self.flops + self.transcendentals

    @property
    def arithmetic_intensity(self) -> float:
        """Arithmetic per byte moved: [`total_flops`][] over [`bytes_accessed`][], in flop/byte.

        Returns `0.0` when no bytes are moved.
        """
        if self.bytes_accessed == 0:
            return 0.0
        return self.total_flops / self.bytes_accessed

    @property
    def attainable_flops(self) -> float | None:
        """The fastest this computation could run on the device, in flop/s.

        This is the minimum of the device's [`DeviceBalance.peak_flops`][] and what its bandwidth
        can sustain at this computation's [`arithmetic_intensity`][], namely
        `arithmetic_intensity * peak_bandwidth`. A compute-bound computation is capped by the
        first, a memory-bound one by the second. `None` if `balance` is `None`.
        """
        if self.balance is None:
            return None
        return min(self.balance.peak_flops, self.arithmetic_intensity * self.balance.peak_bandwidth)

    @property
    def efficiency(self) -> float | None:
        """The attainable throughput as a fraction of peak, in `[0, 1]`.

        A memory-bound computation has a low efficiency *by definition*: it is the fraction of
        the machine's arithmetic units the computation can possibly keep busy, not a measurement of
        how well it is implemented. `None` if no `balance` was measured.
        """
        attainable = self.attainable_flops
        if attainable is None or self.balance is None:
            return None
        return attainable / self.balance.peak_flops

    @property
    def bound(self) -> Bound | None:
        """What limits this computation on the device, or `None` if no `balance` was measured."""
        if self.balance is None:
            return None
        ridge = self.balance.ridge
        if abs(self.arithmetic_intensity - ridge) <= ridge * self.balance.ridge_error:
            return Bound.AMBIGUOUS
        return Bound.MEMORY if self.arithmetic_intensity < ridge else Bound.COMPUTE

    def __str__(self) -> str:
        header = 'Profile' if self.balance is None else f'Profile on {self.balance}'
        if not self.cost_available:
            # every count is zero because the backend declined to report, so nothing derived from
            # them may be shown as a number: it would read as a measurement of a free computation
            flops = intensity = bound = 'unavailable'
        else:
            flops = _format_count(self.flops, 'FLOP')
            intensity = f'{self.arithmetic_intensity:.3g} flop/byte'
            bound = self._format_bound()
        return '\n'.join(
            [
                header,
                f'  flops       {flops}  transcendentals={self.transcendentals:.0f}',
                f'  bytes       {format_bytes(self.bytes_accessed)}',
                f'  intensity   {intensity}',
                f'  bound       {bound}',
                (
                    f'  memory      args={format_bytes(self.argument_bytes)} '
                    f'out={format_bytes(self.output_bytes)} temp={format_bytes(self.temp_bytes)} '
                    f'peak={format_bytes(self.peak_bytes)}'
                ),
            ]
        )

    def _format_bound(self) -> str:
        """Renders the bound line, given that a cost analysis was available."""
        if self.balance is None:
            return 'unknown (no machine balance measured)'
        if self.total_flops == 0:
            # '0.0% of peak' would read as a failure rather than as an absence of arithmetic
            return 'memory (pure data movement, no arithmetic)'
        if self.bound is Bound.AMBIGUOUS:
            return f'{Bound.AMBIGUOUS} (intensity within the ridge measurement error)'
        efficiency = self.efficiency
        assert efficiency is not None and self.attainable_flops is not None  # balance is not None
        return (
            f'{self.bound}-bound, at best {_format_count(self.attainable_flops, "FLOP/s")} '
            f'({efficiency:.1%} of peak)'
        )


def _peak_rate(work: float, times: list[float]) -> tuple[float, float]:
    rates = work / np.asarray(times)
    return float(rates.max()), float(rates.std() / rates.mean())


def _time_repeats(fn: Any, *args: Any) -> list[float]:
    """Returns the wall time of `_REPEATS` calls, in seconds, after one warm-up call."""
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return times


def device_of(tree: Any) -> jax.Device | None:
    """Returns the device a pytree's arrays are committed to, if they agree on one.

    Returns `None` when the tree holds no committed array — no leaves, uncommitted leaves, or
    leaves committed to different devices. That is the signal to fall back to a default rather
    than to trust any one of them.

    Args:
        tree: The pytree to inspect.

    Returns:
        The single device every committed array leaf lives on, or `None`.
    """
    devices = {
        device
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, jax.Array) and leaf.committed
        for device in leaf.devices()
    }
    return devices.pop() if len(devices) == 1 else None


def measure_balance(
    device: jax.Device | None = None, dtype: DTypeLike = jnp.float32
) -> DeviceBalance:
    """Measures a device's peak arithmetic and memory throughput.

    Peak flop/s is measured with a large square matmul ($2n^3$ flops) to keep the arithmetic units
    busy. Peak byte/s is measured by adding two large arrays together ($3n$ elements of traffic).

    Each rate is the best of several repeats, and their relative spread is recorded alongside it.
    Expect a few percent on an idle machine and tens of percent on a contended one.

    The dtype matters: single- and double-precision peak rates differ by up to a factor 64 on
    accelerators. Complex dtypes are measured with their component dtype, and non-floating dtypes
    with float32.

    Args:
        device: The device to measure. Defaults to `jax.devices()[0]`.
        dtype: The dtype to measure the rates for.

    Returns:
        The measured [`DeviceBalance`][].
    """
    if device is None:
        device = jax.devices()[0]
    dtype = _real_float_dtype(dtype)

    with jax.default_device(device):
        n = _MATMUL_SIZE.get(device.platform, _MATMUL_SIZE_DEFAULT)
        a = jnp.ones((n, n), dtype)
        peak_flops, flops_error = _peak_rate(2 * n**3, _time_repeats(jax.jit(jnp.matmul), a, a))

        # two distinct arrays: passing the same buffer twice would let XLA read it once
        x = jnp.ones(_BANDWIDTH_SIZE, dtype)
        y = jnp.zeros(_BANDWIDTH_SIZE, dtype)
        traffic = 3 * x.nbytes  # two reads and one write
        peak_bandwidth, bandwidth_error = _peak_rate(traffic, _time_repeats(jax.jit(jnp.add), x, y))

    return DeviceBalance(
        device.device_kind, dtype, peak_flops, peak_bandwidth, flops_error, bandwidth_error
    )


def _normalize_cost_analysis(cost: Any) -> tuple[dict[str, float], bool]:
    """Reduces the backend-dependent shapes of `cost_analysis()` to a single dict.

    Depending on the backend and the jax version this is `None` (unsupported), a dict, or a list
    holding one dict per computation. Returns the dict and whether one was available at all.
    """
    if isinstance(cost, list):
        cost = cost[0] if cost else None
    if not isinstance(cost, dict):
        return {}, False
    return cost, True


def _normalize_memory_analysis(memory: Any) -> tuple[int, int, int, int]:
    """Extracts `(argument, output, temp, peak)` byte counts, tolerating an absent analysis.

    Like the cost analysis, the memory analysis is optional for a backend to provide.
    """
    if memory is None:
        return 0, 0, 0, 0
    return (
        memory.argument_size_in_bytes,
        memory.output_size_in_bytes,
        memory.temp_size_in_bytes,
        memory.peak_memory_in_bytes,
    )


def profile(
    fn: Any, *args: Any, balance: DeviceBalance | None = None, **kwargs: Any
) -> ProfileReport:
    """Compiles a function and reports its static cost.

    The arguments are passed to [`jax.stages.Lowered`][] and may be concrete arrays or
    [`jax.ShapeDtypeStruct`][] — nothing is executed, so shapes and dtypes are enough.

    Args:
        fn: The function to compile. Must be jittable.
        *args: The positional arguments to lower `fn` with.
        balance: The [`DeviceBalance`][] to evaluate the counts against. Without one, the report
            still carries flops and bytes but cannot say whether the computation is compute- or
            memory-bound.
        **kwargs: The keyword arguments to lower `fn` with.

    Returns:
        The [`ProfileReport`][] for the compiled executable.

    Important:
        Anything `fn` closes over becomes an XLA constant rather than a buffer read: its bytes
        disappear from the count and constant-folding may delete the arithmetic that consumes it
        outright. Pass every array `fn` depends on as an argument instead.

    Examples:
        >>> import jax.numpy as jnp
        >>> report = profile(jnp.matmul, jnp.zeros((64, 64)), jnp.zeros((64, 64)))
        >>> report.flops
        524288.0
    """
    compiled = jax.jit(fn).lower(*args, **kwargs).compile()
    cost, cost_available = _normalize_cost_analysis(compiled.cost_analysis())
    argument_bytes, output_bytes, temp_bytes, peak_bytes = _normalize_memory_analysis(
        compiled.memory_analysis()
    )
    return ProfileReport(
        # a kernel that only moves data has no 'flops' key at all, hence the defaults
        flops=float(cost.get('flops', 0.0)),
        transcendentals=float(cost.get('transcendentals', 0.0)),
        bytes_accessed=float(cost.get('bytes accessed', 0.0)),
        argument_bytes=argument_bytes,
        output_bytes=output_bytes,
        temp_bytes=temp_bytes,
        peak_bytes=peak_bytes,
        balance=balance,
        cost_available=cost_available,
    )
