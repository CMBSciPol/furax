# Data Structures

Furax provides specialized data structures for handling Cosmic Microwave Background (CMB) data, particularly Stokes parameters and sky pixelizations. These structures are built on top of JAX arrays and are designed to be composable, efficient, and mathematically intuitive.

## Stokes Parameters

Stokes parameters describe the polarization state of electromagnetic radiation. In CMB analysis, we work with different combinations of Stokes parameters depending on the analysis requirements.

### Stokes Classes Overview

Furax provides several Stokes parameter classes:

* **StokesI**: Intensity-only measurements
* **StokesQU**: Linear polarization (Q and U parameters)
* **StokesIQU**: Full linear polarization (Intensity + Q + U)
* **StokesIQUV**: Complete Stokes parameters including circular polarization

All Stokes classes inherit from the abstract `Stokes` base class and are JAX PyTree structures, making them compatible with JAX transformations like `jit`, `grad`, and `vmap`.

### Creating Stokes Parameters

```python
import jax.numpy as jnp
from furax.obs.stokes import Stokes

# Create Stokes parameters from arrays
i_data = jnp.ones(100)
q_data = jnp.zeros(100)
u_data = jnp.zeros(100)

# Intensity only
stokes_i = Stokes.from_stokes(i_data)

# Linear polarization
stokes_qu = Stokes.from_stokes(q_data, u_data)

# Full linear polarization
stokes_iqu = Stokes.from_stokes(i_data, q_data, u_data)
```

### Factory Methods

Stokes classes provide convenient factory methods for common initialization patterns:

```python
import jax.random as jr

from furax.obs.stokes import StokesIQU

# Create zero-initialized Stokes parameters
stokes_zero = StokesIQU.zeros(shape=(100,))

# Create ones
stokes_ones = StokesIQU.ones(shape=(100,))

# Create with specific value
stokes_full = StokesIQU.full(shape=(100,), fill_value=2.5)

# Create from random normal distribution
stokes_random = StokesIQU.normal(jr.key(42), shape=(100,))

# Create from uniform distribution
stokes_uniform = StokesIQU.uniform(jr.key(43), shape=(100,), minval=-1, maxval=1)
```

### Arithmetic Operations

Stokes parameters support standard arithmetic operations:

```python
import jax.random as jr

from furax.obs.stokes import StokesIQU

key1, key2 = jr.split(jr.key(0), 2)
stokes1 = StokesIQU.normal(key1, (100,))
stokes2 = StokesIQU.normal(key2, (100,))

# Addition and subtraction
result_add = stokes1 + stokes2
result_diff = abs(stokes1 - stokes2)

# Scalar multiplication / division
result_mul = 2.0 * stokes1
result_div = stokes1 / 3.0

# Element-wise operations maintain Stokes structure
assert isinstance(result_diff, StokesIQU)
```

### Accessing Components

Individual Stokes components can be accessed as properties:

```python
import jax.random as jr

from furax.obs.stokes import StokesIQU

sky = StokesIQU.normal(jr.key(0), (100,))

# Access individual Stokes parameters
intensity = sky.i      # Intensity component
q_parameter = sky.q    # Q polarization parameter
u_parameter = sky.u    # U polarization parameter

# Check available components
print(sky.stokes)      # Returns the Stokes type string, e.g., 'IQU'
```

## Sky Landscapes

Landscapes represent sky pixelizations and provide the spatial structure for CMB maps. They handle coordinate systems, pixelization schemes, and spatial operations.

### Landscape Classes

* **HealpixLandscape**: HEALPix pixelization scheme
* **StokesLandscape**: Multi-dimensional Stokes parameter maps

### HealpixLandscape

The most commonly used landscape for CMB analysis is based on the HEALPix pixelization:

```python
from furax.obs.landscapes import HealpixLandscape

# Create a HEALPix landscape for polarization analysis
landscape = HealpixLandscape(nside=32, stokes='QU')

print(f'Number of pixels: {landscape.shape[0]}')
print(f'Stokes parameters: {landscape.stokes}')
print(f'Total size (Q plus U): {landscape.size}')
```

### Landscape Factories

Landscapes provide methods for generating data:

```python
import jax.random as jr

from furax.obs.landscapes import HealpixLandscape

landscape = HealpixLandscape(nside=64, stokes='IQU')
key = jr.key(42)

# Generate different types of random data

# Gaussian random field
normal_map = landscape.normal(key)
print(f'Data structure: {normal_map.structure}')

# Uniform random field
uniform_map = landscape.uniform(key, minval=-1, maxval=1)

# Constant map
constant_map = landscape.full(fill_value=1.73)

# Zero map
zero_map = landscape.zeros()
```

### Working with Real Data

Landscapes can be used with real CMB data:

```python
import healpy as hp
from furax.obs.landscapes import HealpixLandscape
from furax.obs.stokes import Stokes

# Load real CMB map (example)
cmb_map = hp.read_map('cmb_data.fits', field=[0, 1, 2])  # I, Q, U

# Create landscape matching the data
nside = hp.get_nside(cmb_map)
landscape = HealpixLandscape(nside=nside, stokes='IQU')

# Convert to Furax Stokes structure
stokes_data = Stokes.from_stokes(*cmb_map)

# Verify compatibility
assert stokes_data.shape == landscape.shape
```

## Integration with Linear Operators

The real power of Furax data structures comes from their integration with linear operators:

```python
import jax.random as jr

from furax import DiagonalOperator
from furax.obs.landscapes import HealpixLandscape

sky_key, weight_key = jr.split(jr.key(0), 2)

# Create landscape and data
landscape = HealpixLandscape(nside=32, stokes='QU')
data = landscape.normal(sky_key)

# Create an operator that applies a same weight to the Q and U parameters
weights = 1 + jr.normal(weight_key, landscape.shape) / 100
weight_operator = DiagonalOperator(weights, in_structure=landscape.structure)

# Apply operator to data
weighted_data = weight_operator(data)

# The result maintains the same Stokes structure
print(f'Input type: {data.structure}')
print(f'Output type: {weighted_data.structure}')
```

## Advanced Usage

### JAX Transformations

Since all data structures are JAX PyTrees, they work seamlessly with JAX transformations:

```python
import jax
import jax.numpy as jnp
import jax.random as jr

from furax.obs.landscapes import HealpixLandscape

noise_levels = jnp.array([0.1, 0.2, 0.3])
sky_key, *noise_keys = jr.split(jr.key(0), 1 + len(noise_levels))

landscape = HealpixLandscape(nside=16, stokes='QU')
sky = landscape.normal(jr.key(0))
keys = jr.split(jr.key(0), 3)

@jax.jit
def add_noise(key, stokes_map, noise_level):
    return stokes_map + noise_level * landscape.normal(key)

# Vectorize over different noise levels
add_noise_vmap = jax.vmap(add_noise, in_axes=(0, None, 0), out_axes=0)

# Process with different noise levels
noisy_sky = add_noise_vmap(keys, sky, noise_levels)
```

### Memory Efficiency

For large-scale analysis, consider memory usage:

```python
import jax
import jax.random as jr

from furax.obs.landscapes import HealpixLandscape

key = jr.key(0)

# For very high resolution maps
landscape_highres = HealpixLandscape(nside=2048, stokes='IQU')
print(f"Memory per map: ~{landscape_highres.nbytes / 2**20:.1f} MiB")

# Use appropriate precision
float64_map = landscape_highres.normal(key)
with jax.experimental.disable_x64():
    float32_map = landscape_highres.normal(key)
```

## Best Practices

1. **Choose appropriate Stokes combinations**: Use `StokesI` for intensity-only analysis, `StokesQU` for polarization-only, etc.

2. **Match landscape resolution to your analysis**: Higher `nside` values provide more spatial resolution but require more memory.

3. **Leverage JAX transformations**: Use `jit`, `vmap`, and `grad` for performance and automatic differentiation.

4. **Maintain data structure consistency**: Operations between Stokes parameters and operators preserve the underlying structure.

5. **Use factory methods**: Prefer `landscape.normal(key)` over manual array construction for consistency.
