# Getting Started

Welcome to Furax! This guide will help you get up and running with CMB analysis using Furax's composable linear operators and specialized data structures.

## Installation

### Basic Installation

Install Furax using pip:

```bash
pip install furax
```

### Development Installation

For development or to access the latest features:

```bash
git clone https://github.com/your-org/furax.git
cd furax
pip install -e .[dev]
```

### Component Separation Features

For advanced component separation capabilities:

```bash
pip install -e .[comp_sep]
```

This includes additional dependencies like PySM3 for foreground modeling.

### Dependencies

Furax relies on the JAX ecosystem and scientific Python packages:

- **Core**: JAX
- **Astronomy**: jax-healpy, astropy
- **Development**: pytest, pre-commit, ruff, mypy

## First Steps

Enable 64-bit precision for better numerical accuracy:

```python
import jax
jax.config.update('jax_enable_x64', True)
```

### Create Your First Sky Map

```python
import jax.random as jr

from furax.obs.landscapes import HealpixLandscape

# Create a HEALPix landscape for polarization analysis
landscape = HealpixLandscape(nside=32, stokes='IQU')

# Generate a random CMB-like sky
cmb_map = landscape.normal(jr.key(42))

print(f'Map shape: {cmb_map.shape}')
print(f'Stokes parameters: {cmb_map.stokes}')
print(f'Number of pixels: {landscape.shape[0]}')
```

### Basic Linear Operators

Furax provides composable linear operators that can be combined through addition, composition of block assembly. The primary interest of these operators is that they rely on a sparse representation of the underlying matrices.

```python
import jax.numpy as jnp
import jax.random as jr

from furax import DiagonalOperator
from furax.tree import as_structure
from furax.obs.landscapes import HealpixLandscape

landscape = HealpixLandscape(nside=32, stokes='IQU')
cmb_map = landscape.normal(jr.key(42))
n_pixel = landscape.shape[0]

# Create a noise weighting operator
noise_weights = DiagonalOperator(1.0 / jnp.full(n_pixel, jnp.sqrt(n_pixel)), in_structure=landscape.structure)

# Apply the weights to the I, Q and U Stokes parameters of the map
weighted_map = noise_weights(cmb_map)

print(f'Input type: {as_structure(cmb_map)}')
print(f'Output type: {as_structure(weighted_map)}')
```

### Operator Composition

The power of Furax comes from composable operators:

```python
import jax.numpy as jnp
import jax.random as jr

from furax import BlockDiagonalOperator, DiagonalOperator
from furax.obs.landscapes import HealpixLandscape
from furax.obs.stokes import StokesIQU

landscape = HealpixLandscape(nside=32, stokes='IQU')
cmb_map = landscape.normal(jr.key(42))
n_pixel = landscape.shape[0]

# Create a component-wise processing
i_processor = DiagonalOperator(1.0 * jnp.ones(n_pixel))  # No change to I
q_processor = DiagonalOperator(2.0 * jnp.ones(n_pixel))  # Amplify Q
u_processor = DiagonalOperator(0.5 * jnp.ones(n_pixel))  # Reduce U

# Combine into block diagonal operator
component_processor = BlockDiagonalOperator(StokesIQU(i_processor, q_processor, u_processor))

# The noise weights apply the same diagonal matrix to I, Q and U
noise_weights = DiagonalOperator(1.0 / jnp.full(n_pixel, jnp.sqrt(n_pixel)), in_structure=landscape.structure)

# Compose with noise weighting
full_pipeline = component_processor @ noise_weights

# Apply the full pipeline
processed_map = full_pipeline(cmb_map)

print(f'Pipeline applied successfully!')
```

## Working with Real Data

### Reconstruction problem

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from furax import HomothetyOperator, IndexOperator
from furax.tree import as_structure

n_pixel = 10
pixels = jnp.arange(n_pixel, dtype=jnp.int32)
obs_key, map_key, noise_key = jr.split(jr.key(0), 3)

observed_pixels = jnp.concatenate([jr.permutation(key, pixels) for key in jr.split(obs_key, 100)])
actual_map = jr.normal(map_key, (n_pixel,))
σ_noise = 0.01
noise = jr.normal(noise_key, observed_pixels.shape) * σ_noise

acquisition_op = IndexOperator(observed_pixels, in_structure=as_structure(actual_map))
observed_values = acquisition_op(actual_map) + noise

noise_op = HomothetyOperator(σ_noise ** 2, in_structure=as_structure(observed_values))

ml = (acquisition_op.T @ noise_op.I @ acquisition_op).I @ acquisition_op.T @ noise_op.I

# Using default setup (using CG)
maximum_likelihood_map = ml(observed_values)
print('Actual map:', actual_map)
print('Reconstructed map:', maximum_likelihood_map)
print('Difference:', abs(actual_map - maximum_likelihood_map))

# Use high-precision solver for critical calculations
solver = lx.CG(rtol=1e-10, atol=1e-10, max_steps=2000)
high_precision_ml = (acquisition_op.T @ noise_op.I @ acquisition_op).I(solver=solver) @ acquisition_op.T @ noise_op.I

high_precision_map = high_precision_ml(observed_values)
print('Difference:', abs(actual_map - high_precision_map))
```

### Pixel Masking

```python
import jax.numpy as jnp
import jax.random as jr
import jax_healpy as hp

from furax import IndexOperator
from furax.obs.landscapes import HealpixLandscape
from furax.tree import as_structure

GALACTIC_MAX_LATITUDE = 5.  # degrees

landscape = HealpixLandscape(nside=128, stokes='IQU')
n_pixel = landscape.shape[0]
pixels = jnp.arange(n_pixel, dtype=jnp.int32)
lon, lat = hp.pix2ang(landscape.nside, pixels, lonlat=True)
good_pixels = abs(lat) > GALACTIC_MAX_LATITUDE

# Create a galactic plane mask (simplified)
mask_operator = IndexOperator(jnp.where(good_pixels), in_structure=landscape.structure)

# Apply mask
cmb_map = landscape.normal(jr.key(0))
masked_map = mask_operator(cmb_map)
print(f'Input map: {as_structure(cmb_map)}')
print(f'Output map: {as_structure(masked_map)}')
```

### Frequency Analysis

```python
import jax.numpy as jnp
import jax.random as jr

from furax import IndexOperator
from furax.obs.landscapes import HealpixLandscape
from furax.tree import as_structure

# Multi-frequency analysis setup
frequencies = jnp.array([70., 150., 353.])  # GHz
landscape = HealpixLandscape(nside=128, stokes='IQU')
n_pixel = landscape.shape[0]

# Create multi-frequency landscape
obs_key, *keys = jr.split(jr.key(0), len(frequencies) + 1)
freq_maps = [landscape.normal(key) for key in keys]

pixels = jr.randint(obs_key, (100,), 0, n_pixel - 1)
projection = IndexOperator(pixels, in_structure=landscape.structure)

# get the observed pixels (noiseless)
tod = projection(freq_maps)

# The tod is a list of StokesIQU
print(f'Multi-frequency tod structure: {as_structure(tod)}')
```

## Error Handling and Debugging

### Check Operator Properties

```python
# Inspect operator properties
op = ...
print(f'Operator is symmetric: {op.is_symmetric}')
print(f'Operator is positive definite: {op.is_positive_semidefinite}')
print(f'Operator input structure: {op.in_structure}')
print(f'Operator output structure: {op.out_structure}')
```

### Matrix Visualization

For small problems, visualize operators as matrices:

```python
import jax.numpy as jnp

from furax import DiagonalOperator
from furax.obs.landscapes import HealpixLandscape

# Only for small operators!
small_landscape = HealpixLandscape(nside=2, stokes='I')  # 48 pixels
small_weights = DiagonalOperator(1. + jnp.arange(small_landscape.shape[0]))

# Convert to explicit matrix for debugging
weight_matrix = small_weights.as_matrix()
print(f'Weight matrix shape: {weight_matrix.shape}')
print('Diagonal elements:', jnp.diag(weight_matrix))
```

## Performance Tips

### Use JAX Transformations

```python
import jax
import jax.random as jr

from furax import DiagonalOperator
from furax.obs.landscapes import HealpixLandscape

# JIT compile for repeated operations
@jax.jit
def process_many_maps(operator, maps):
    return jax.vmap(lambda m: operator(m))(maps)

batch_size = 10
landscape = HealpixLandscape(nside=128, stokes='IQU')
op_key = jr.key(0)
map_keys = jr.split(jr.key(1), batch_size)
op = DiagonalOperator(1 + 0.01 * jr.normal(op_key, landscape.shape))

# Generate batch of maps
map_batch = jax.vmap(landscape.normal)(map_keys)

# Process batch efficiently
processed_batch = process_many_maps(op, map_batch)
print(f'Processed {batch_size} maps in batch: {processed_batch.structure}')
```

### Memory Management

```python
import jax.numpy as jnp
import jax.random as jr

from furax import DiagonalOperator
from furax.obs.landscapes import HealpixLandscape

# For large problems, avoid creating explicit matrices
landscape = HealpixLandscape(nside=256, stokes='IQU')  # ~200k parameters

# Good: matrix-free operations
large_weights = DiagonalOperator(jnp.ones(landscape.shape[0]), in_structure=landscape.structure)
large_map = landscape.random(jr.key(0))  # Zero map to avoid memory for random data
result = large_weights(large_map)

# Avoid: large_weights.as_matrix() - would use ~160GB for float64!
print(f'Matrix-free operation completed for {landscape.size} parameters')
```

## Next Steps

Now that you've learned the basics:

1. **Data Structures**: Explore [data_structures.md](data_structures.md) for advanced Stokes parameter usage
2. **Linear Operators**: Learn about operator composition in [operators.md](operators.md)
3. **Examples**: Try the [component_separation.md](../examples/component_separation.md) and [mapmaking.md](../examples/mapmaking.md) tutorials
4. **API Reference**: Browse the complete API reference for all available functions

Happy analyzing!
