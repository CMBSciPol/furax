"""Static cost model for compiled computations.

Reports the flops, memory traffic and buffer sizes of a computation without running it. The
computation is lowered and compiled ahead of time, and XLA's estimates are read off the resulting
executable with [`jax.stages.Compiled.cost_analysis`][] (flops, bytes accessed) and
[`jax.stages.Compiled.memory_analysis`][] (argument / output / temporary buffer sizes). Shapes and
dtypes are therefore enough: no input needs to be allocated.
"""

import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

__all__ = [
    'DeviceBalance',
    'ProfileReport',
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

_BALANCE_CACHE: dict[tuple[jax.Device, np.dtype[np.floating[Any]]], 'DeviceBalance'] = {}


def format_bytes(n: float) -> str:
    """Formats a byte count with a binary unit suffix.

    Args:
        n: The number of bytes.

    Returns:
        The byte count, rounded to two decimals and suffixed with a binary unit.

    Examples:
        >>> format_bytes(1536)
        '1.50KiB'
    """
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if n < 1024:
            return f'{n:.2f}{unit}'
        n /= 1024
    return f'{n:.2f}PiB'


def _format_count(n: float, unit: str) -> str:
    """Formats a count with a decimal SI prefix, e.g. ``_format_count(2.1e9, 'FLOP')``."""
    for prefix in ('', 'K', 'M', 'G', 'T'):
        if n < 1000:
            return f'{n:.2f} {prefix}{unit}'
        n /= 1000
    return f'{n:.2f} P{unit}'


def _real_float_dtype(dtype: DTypeLike) -> np.dtype[np.floating[Any]]:
    """Maps a dtype onto the real floating-point dtype whose peak rates should be measured.

    Complex dtypes are mapped to their component dtype and non-inexact dtypes to float32, since the
    microbenchmarks measure floating-point throughput. float64 is downgraded when x64 is disabled,
    where float64 arrays cannot be created at all.
    """
    given = np.dtype(dtype)
    if np.issubdtype(given, np.complexfloating):
        result = np.dtype(given.type(0).real.dtype)
    elif np.issubdtype(given, np.floating):
        result = np.dtype(given)
    else:
        result = np.dtype(np.float32)
    if result == np.float64 and not jax.config.jax_enable_x64:
        result = np.dtype(np.float32)
    return result


@dataclass(frozen=True)
class DeviceBalance:
    """Measured peak throughputs of a device, for one dtype.

    A device performs only so many flops per second and moves only so many bytes per second. Their
    ratio, [`ridge`][], is how much arithmetic it must be given per byte to keep its arithmetic
    units fed — typically tens of flops per byte.

    Compare it to a computation's [`ProfileReport.arithmetic_intensity`][], the flops it performs
    per byte it touches. Below the ridge the computation is *memory-bound*: the units idle waiting
    for data. Above the ridge it is *compute-bound*: the data arrives faster than the units consume
    it, and the arithmetic sets the pace.

    A matrix-vector product reads a whole matrix to do only two flops per element, so it is
    typically memory-bound on all devices; a large matrix-matrix product reuses each element many
    times, and is compute-bound on most devices.

    Attributes:
        device_kind: The `device_kind` of the [`jax.Device`][], e.g. `'NVIDIA H100 80GB HBM3'`.
        dtype: The dtype the rates were measured with.
        peak_flops: Measured peak arithmetic throughput, in flop/s.
        peak_bandwidth: Measured peak memory throughput, in byte/s.
    """

    device_kind: str
    dtype: np.dtype[np.floating[Any]]
    peak_flops: float
    peak_bandwidth: float

    @property
    def ridge(self) -> float:
        """The ridge point in flop/byte: [`peak_flops`][] divided by [`peak_bandwidth`][]."""
        return self.peak_flops / self.peak_bandwidth

    def __str__(self) -> str:
        return (
            f'{self.device_kind} ({self.dtype.name}) '
            f'peak={_format_count(self.peak_flops, "FLOP/s")} '
            f'bandwidth={format_bytes(self.peak_bandwidth)}/s '
            f'ridge={self.ridge:.3g} flop/byte'
        )


@dataclass(frozen=True)
class ProfileReport:
    """The static cost of one compiled computation.

    This is the result of a call to [`profile`][], or [`AbstractLinearOperator.profile`]
    [furax.core.AbstractLinearOperator.profile] for an operator.

    Warning:
        `flops` and `bytes_accessed` are estimates from XLA's static cost model, not measurements
        from hardware counters. JAX documents that model as a debugging aid whose structure "may be
        inconsistent across versions of JAX and jaxlib, or even across invocations", and the
        individual counters are undocumented XLA internals. Read these numbers, and everything
        derived from them, as indications rather than facts.

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

        Two ceilings apply and the lower one wins: the device's [`DeviceBalance.peak_flops`][], and
        what its bandwidth can sustain at this computation's [`arithmetic_intensity`][], which is
        `arithmetic_intensity * peak_bandwidth`. A compute-bound computation is capped by the
        first, a memory-bound one by the second. `None` if `balance` is `None`.
        """
        if self.balance is None:
            return None
        return min(self.balance.peak_flops, self.arithmetic_intensity * self.balance.peak_bandwidth)

    @property
    def efficiency(self) -> float | None:
        """The attainable throughput as a fraction of peak, in `[0, 1]`.

        A memory-bound computation has a low efficiency *by construction*: it is the fraction of
        the machine's arithmetic units the computation can possibly keep busy, not a measurement of
        how well it is implemented. `None` if no `balance` was measured.
        """
        attainable = self.attainable_flops
        if attainable is None or self.balance is None:
            return None
        return attainable / self.balance.peak_flops

    @property
    def is_memory_bound(self) -> bool | None:
        """Whether the arithmetic intensity is below the machine's ridge point.

        `None` if no `balance` was measured.
        """
        if self.balance is None:
            return None
        return self.arithmetic_intensity < self.balance.ridge

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
        efficiency = self.efficiency
        assert efficiency is not None and self.attainable_flops is not None  # balance is not None
        bound = 'memory' if self.is_memory_bound else 'compute'
        return (
            f'{bound}-bound, at best {_format_count(self.attainable_flops, "FLOP/s")} '
            f'({efficiency:.1%} of peak)'
        )


def _time_best(fn: Any, *args: Any) -> float:
    """Returns the shortest wall time of `_REPEATS` calls, in seconds, after one warm-up call."""
    jax.block_until_ready(fn(*args))
    best = float('inf')
    for _ in range(_REPEATS):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - start)
    return best


def measure_balance(
    device: jax.Device | None = None, dtype: DTypeLike = jnp.float32
) -> DeviceBalance:
    """Measures a device's peak arithmetic and memory throughput.

    Peak flop/s is measured with a large square matmul ($2n^3$ flops), which has enough data reuse
    to keep the arithmetic units busy. Peak byte/s is measured by adding two arrays far larger than
    any cache ($3n$ elements of traffic: two read, one written), which has none. Both take the best
    of several runs after a warm-up. Results are cached per `(device, dtype)`, so repeated calls
    are free.

    The dtype matters: single- and double-precision peak rates differ by up to a factor 64 on
    accelerators. Complex dtypes are measured with their component dtype, and integer dtypes with
    float32.

    Args:
        device: The device to measure. Defaults to `jax.devices()[0]`.
        dtype: The dtype to measure the rates for.

    Returns:
        The measured [`DeviceBalance`][].

    Warning:
        This runs actual computations and takes on the order of a second the first time it is
        called for a given `(device, dtype)`. Pass `measure=False` to
        [`AbstractLinearOperator.profile`][furax.core.AbstractLinearOperator.profile] to skip it.
    """
    if device is None:
        device = jax.devices()[0]
    dtype = _real_float_dtype(dtype)

    key = (device, dtype)
    if (cached := _BALANCE_CACHE.get(key)) is not None:
        return cached

    with jax.default_device(device):
        n = _MATMUL_SIZE.get(device.platform, _MATMUL_SIZE_DEFAULT)
        a = jnp.ones((n, n), dtype)
        seconds = _time_best(jax.jit(jnp.matmul), a, a)
        peak_flops = 2 * n**3 / seconds

        # two distinct arrays: passing the same buffer twice would let XLA read it once
        x = jnp.ones(_BANDWIDTH_SIZE, dtype)
        y = jnp.zeros(_BANDWIDTH_SIZE, dtype)
        seconds = _time_best(jax.jit(jnp.add), x, y)
        peak_bandwidth = 3 * x.nbytes / seconds  # two reads and one write

    balance = DeviceBalance(device.device_kind, dtype, peak_flops, peak_bandwidth)
    _BALANCE_CACHE[key] = balance
    return balance


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
        outright. Pass every array `fn` depends on as an argument instead. This is why
        [`AbstractLinearOperator.profile`][furax.core.AbstractLinearOperator.profile] lowers
        `lambda op, x: op.mv(x)` with the operator as an argument, rather than lowering `op.mv`.

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
