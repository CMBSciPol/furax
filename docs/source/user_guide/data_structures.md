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
import jax
import jax.numpy as jnp
from furax.obs import Stokes

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
from furax.obs import StokesIQU

# Create zero-initialized Stokes parameters
stokes_zero = StokesIQU.zeros(shape=(100,))

# Create ones
stokes_ones = StokesIQU.ones(shape=(100,))

# Create with specific value
stokes_full = StokesIQU.full(shape=(100,), fill_value=2.5)

# Create from random normal distribution
key = jax.random.PRNGKey(42)
stokes_random = StokesIQU.normal(key, shape=(100,))

# Create from uniform distribution
stokes_uniform = StokesIQU.uniform(key, shape=(100,), minval=-1, maxval=1)
```

### Arithmetic Operations

Stokes parameters support standard arithmetic operations:

```python
stokes1 = StokesIQU.normal(jax.random.PRNGKey(0), (100,))
stokes2 = StokesIQU.normal(jax.random.PRNGKey(1), (100,))

# Addition and subtraction
result_add = stokes1 + stokes2
result_sub = stokes1 - stokes2

# Scalar multiplication
result_mul = 2.0 * stokes1
result_div = stokes1 / 3.0

# Element-wise operations maintain Stokes structure
assert isinstance(result_add, StokesIQU)
```

### Accessing Components

Individual Stokes components can be accessed as properties:

```python
stokes = StokesIQU.normal(jax.random.PRNGKey(0), (100,))

# Access individual components
intensity = stokes.I      # Intensity component
q_param = stokes.Q        # Q polarization parameter
u_param = stokes.U        # U polarization parameter

# Check available components
print(stokes.stokes)      # Returns the Stokes type string, e.g., 'IQU'
```

## Sky Landscapes

Landscapes represent sky pixelizations and provide the spatial structure for CMB maps. They handle coordinate systems, pixelization schemes, and spatial operations.

### Landscape Classes

* **HealpixLandscape**: HEALPix pixelization scheme
* **StokesLandscape**: Multi-dimensional Stokes parameter maps

### HealpixLandscape

The most commonly used landscape for CMB analysis, based on the HEALPix pixelization:

```python
from furax.obs import HealpixLandscape

# Create a HEALPix landscape for polarization analysis
landscape = HealpixLandscape(nside=32, stokes='QU')

print(f"Number of pixels: {landscape.npix}")
print(f"Stokes parameters: {landscape.stokes}")
print(f"Total size: {landscape.size}")

# Generate random data matching the landscape
key = jax.random.PRNGKey(42)
map_data = landscape.normal(key)

print(f"Data shape: {map_data.shape}")
print(f"Data type: {type(map_data)}")
```

### Landscape Operations

Landscapes provide methods for generating data and performing spatial operations:

```python
landscape = HealpixLandscape(nside=64, stokes='IQU')

# Generate different types of random data
key = jax.random.PRNGKey(123)

# Gaussian random field
gaussian_map = landscape.normal(key)

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
from furax.obs import HealpixLandscape, Stokes

# Load real CMB map (example)
# cmb_map = hp.read_map('cmb_data.fits', field=[0, 1, 2])  # I, Q, U

# Create landscape matching the data
nside = 512  # Adjust based on your data
landscape = HealpixLandscape(nside=nside, stokes='IQU')

# Convert to Furax Stokes structure
# stokes_data = Stokes.from_stokes(*cmb_map)

# Verify compatibility
# assert stokes_data.shape[0] == landscape.npix
```

## Integration with Linear Operators

The real power of Furax data structures comes from their integration with linear operators:

```python
from furax.core import DiagonalOperator
from furax.obs import HealpixLandscape

# Create landscape and data
landscape = HealpixLandscape(nside=32, stokes='QU')
stokes_data = landscape.normal(jax.random.PRNGKey(0))

# Create a weighting operator
weights = jnp.ones(landscape.size)
weight_operator = DiagonalOperator(weights)

# Apply operator to data
weighted_data = weight_operator @ stokes_data

# The result maintains the same Stokes structure
print(f"Input type: {type(stokes_data)}")
print(f"Output type: {type(weighted_data)}")
```

## Advanced Usage

### JAX Transformations

Since all data structures are JAX PyTrees, they work seamlessly with JAX transformations:

```python
from functools import partial

def process_map(stokes_map, noise_level):
    return stokes_map + noise_level * StokesQU.normal(
        jax.random.PRNGKey(0), stokes_map.shape
    )

# JIT compile the function
process_map_jit = jax.jit(process_map)

# Vectorize over different noise levels
process_map_vmap = jax.vmap(
    partial(process_map, stokes_map),
    in_axes=(0,)
)

landscape = HealpixLandscape(nside=16, stokes='QU')
test_map = landscape.normal(jax.random.PRNGKey(1))
noise_levels = jnp.array([0.1, 0.2, 0.3])

# Process with different noise levels
results = process_map_vmap(noise_levels)
```

### Memory Efficiency

For large-scale analysis, consider memory usage:

```python
# For very high resolution maps
landscape_highres = HealpixLandscape(nside=2048, stokes='IQU')
print(f"Memory per map: ~{landscape_highres.size * 4 / 1e6:.1f} MB")

# Use appropriate precision
import jax
with jax.config.context(x64_enable=False):  # Use float32
    efficient_map = landscape_highres.normal(jax.random.PRNGKey(0))
```

## Best Practices

1. **Choose appropriate Stokes combinations**: Use `StokesI` for intensity-only analysis, `StokesQU` for polarization-only, etc.

2. **Match landscape resolution to your analysis**: Higher `nside` values provide more spatial resolution but require more memory.

3. **Leverage JAX transformations**: Use `jit`, `vmap`, and `grad` for performance and automatic differentiation.

4. **Maintain data structure consistency**: Operations between Stokes parameters and operators preserve the underlying structure.

5. **Use factory methods**: Prefer `landscape.normal(key)` over manual array construction for consistency.
