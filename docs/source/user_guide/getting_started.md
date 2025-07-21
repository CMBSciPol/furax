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

- **Core**: JAX, Lineax, NumPy
- **Astronomy**: HealPy, AstroPy, jax-healpy
- **Development**: pytest, ruff, mypy

## First Steps

### Import Furax

```python
import jax
import jax.numpy as jnp
import furax
```

Enable 64-bit precision for better numerical accuracy:

```python
jax.config.update('jax_enable_x64', True)
```

### Create Your First Sky Map

```python
from furax.obs import HealpixLandscape

# Create a HEALPix landscape for polarization analysis
landscape = HealpixLandscape(nside=32, stokes='IQU')

# Generate a random CMB-like sky
key = jax.random.PRNGKey(42)
cmb_map = landscape.normal(key)

print(f"Map shape: {cmb_map.shape}")
print(f"Stokes parameters: {cmb_map.stokes}")
print(f"Number of pixels: {landscape.npix}")
```

### Basic Linear Operators

```python
from furax.core import DiagonalOperator

# Create a noise weighting operator
noise_variance = jnp.ones(landscape.size)
noise_weights = DiagonalOperator(1.0 / noise_variance)

# Apply to your map
weighted_map = noise_weights @ cmb_map

print(f"Input type: {type(cmb_map)}")
print(f"Output type: {type(weighted_map)}")
```

### Operator Composition

The power of Furax comes from composable operators:

```python
from furax.core import BlockDiagonalOperator

# Create component-wise processing
n_pix = landscape.npix

# Different processing for I, Q, U
i_processor = DiagonalOperator(1.0 * jnp.ones(n_pix))      # No change to I
q_processor = DiagonalOperator(2.0 * jnp.ones(n_pix))      # Amplify Q
u_processor = DiagonalOperator(0.5 * jnp.ones(n_pix))      # Reduce U

# Combine into block diagonal operator
component_processor = BlockDiagonalOperator([
    i_processor, q_processor, u_processor
])

# Compose with noise weighting
full_pipeline = component_processor @ noise_weights

# Apply the full pipeline
processed_map = full_pipeline @ cmb_map

print(f"Pipeline applied successfully!")
```

## Working with Real Data

### Loading HEALPix Maps

```python
import healpy as hp
from furax.obs import Stokes

# Load a real CMB map (example with Planck data)
# planck_map = hp.read_map('planck_cmb.fits', field=[0, 1, 2])  # I, Q, U

# For this example, simulate Planck-like data
nside_planck = 512
landscape_hires = HealpixLandscape(nside=nside_planck, stokes='IQU')
simulated_planck = landscape_hires.normal(jax.random.PRNGKey(100))

print(f"High-resolution map: {landscape_hires.npix} pixels")
```

### Configuration Management

Control solver settings with configuration contexts:

```python
import lineax as lx
from furax import Config

# Use high-precision solver for critical calculations
with Config(solver=lx.CG(rtol=1e-10, max_steps=2000)):
    precise_result = full_pipeline @ cmb_map

# Default solver for routine operations
standard_result = full_pipeline @ cmb_map

print("Configuration contexts allow flexible solver control")
```

## Common Patterns

### Pixel Masking

```python
from furax.core import IndexOperator

# Create a galactic plane mask (simplified)
good_pixels = jnp.arange(n_pix)[::2]  # Keep every other pixel
mask_operator = IndexOperator(good_pixels, input_size=landscape.size)

# Apply mask
masked_data = mask_operator @ cmb_map
print(f"Masked data size: {masked_data.shape}")
```

### Frequency Analysis

```python
# Multi-frequency analysis setup
frequencies = jnp.array([30., 44., 70., 100., 143., 217., 353.])  # GHz
n_freq = len(frequencies)

# Create multi-frequency landscape
freq_maps = []
for i, freq in enumerate(frequencies):
    freq_key = jax.random.PRNGKey(200 + i)
    freq_map = landscape.normal(freq_key)
    freq_maps.append(freq_map)

# Stack frequency maps
multi_freq_data = jnp.stack([fmap.flatten() for fmap in freq_maps])
print(f"Multi-frequency data shape: {multi_freq_data.shape}")
```

## Error Handling and Debugging

### Check Operator Properties

```python
# Inspect operator properties
print(f"Operator is symmetric: {noise_weights.symmetric}")
print(f"Operator is positive definite: {noise_weights.positive_semidefinite}")
print(f"Operator shape: {noise_weights.shape}")
```

### Matrix Visualization

For small problems, visualize operators as matrices:

```python
# Only for small operators!
small_landscape = HealpixLandscape(nside=2, stokes='I')  # 12 pixels
small_weights = DiagonalOperator(jnp.arange(1., 13.))

# Convert to explicit matrix for debugging
weight_matrix = small_weights.as_matrix()
print(f"Weight matrix shape: {weight_matrix.shape}")
print("Diagonal elements:", jnp.diag(weight_matrix))
```

## Performance Tips

### Use JAX Transformations

```python
# JIT compile for repeated operations
@jax.jit
def process_many_maps(operator, maps):
    return jax.vmap(lambda m: operator @ m)(maps)

# Generate batch of maps
keys = jax.random.split(jax.random.PRNGKey(500), 10)
map_batch = jax.vmap(lambda k: landscape.normal(k))(keys)

# Process batch efficiently
processed_batch = process_many_maps(noise_weights, map_batch)
print(f"Processed {len(processed_batch)} maps in batch")
```

### Memory Management

```python
# For large problems, avoid creating explicit matrices
large_landscape = HealpixLandscape(nside=128, stokes='IQU')  # ~200k parameters

# Good: matrix-free operations
large_weights = DiagonalOperator(jnp.ones(large_landscape.size))
large_map = large_landscape.zeros()  # Zero map to avoid memory for random data
result = large_weights @ large_map

# Avoid: large_weights.as_matrix() - would use ~160GB for float64!
print(f"Matrix-free operation completed for {large_landscape.size} parameters")
```

## Next Steps

Now that you've learned the basics:

1. **Data Structures**: Explore [data_structures.md](data_structures.md) for advanced Stokes parameter usage
2. **Linear Operators**: Learn about operator composition in [operators.md](operators.md)
3. **Examples**: Try the [component_separation.md](../examples/component_separation.md) and [mapmaking.md](../examples/mapmaking.md) tutorials
4. **API Reference**: Browse the complete API reference for all available functions

Happy analyzing!
