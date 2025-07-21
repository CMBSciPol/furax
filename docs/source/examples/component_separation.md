# CMB Component Separation

This example demonstrates how to use Furax for Cosmic Microwave Background (CMB) component separation, a key analysis technique for extracting the primordial CMB signal from foreground contamination.

## Overview

CMB component separation aims to disentangle the cosmological CMB signal from astrophysical foregrounds such as:

- **Thermal dust emission**: Dominant at high frequencies
- **Synchrotron radiation**: Dominant at low frequencies  
- **Free-free emission**: Relatively flat spectrum
- **Spinning dust**: Peaks around 20-30 GHz

Furax provides spectral energy density (SED) operators that model the frequency dependence of these components, enabling linear algebra approaches to component separation.

## Basic Component Separation

### Setting Up the Problem

```python
import jax
import jax.numpy as jnp
from furax.obs import HealpixLandscape, StokesIQU
from furax.obs.operators import CMBOperator, DustOperator, SynchrotronOperator
from furax.obs.operators import MixingMatrixOperator
from furax.core import DiagonalOperator, BlockDiagonalOperator

# Define observation frequencies (GHz)
frequencies = jnp.array([30., 44., 70., 100., 143., 217., 353.])
n_freq = len(frequencies)

# Create sky landscape (low resolution for example)
nside = 32
landscape = HealpixLandscape(nside=nside, stokes='IQU')
n_pix = landscape.npix
```

### Create Spectral Components

```python
# Create SED operators for each component
cmb_sed = CMBOperator(frequencies)
dust_sed = DustOperator(frequencies, beta=1.54, T_dust=20.0)  # Typical values
sync_sed = SynchrotronOperator(frequencies, beta=-3.1)        # Typical spectral index

print(f"CMB SED shape: {cmb_sed.shape}")
print(f"Dust SED shape: {dust_sed.shape}")  
print(f"Synchrotron SED shape: {sync_sed.shape}")
```

### Build the Mixing Matrix

The mixing matrix relates the observed data to the underlying components:

```python
# Combine SED operators into a mixing matrix
# Each column represents one component's frequency dependence
mixing_matrix = MixingMatrixOperator([cmb_sed, dust_sed, sync_sed])

print(f"Mixing matrix shape: {mixing_matrix.shape}")
print(f"Components: CMB, Dust, Synchrotron")
```

### Simulate Multi-frequency Data  

```python
# Generate true sky components
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 3)

# True component amplitudes (IQU for each pixel)
cmb_true = landscape.normal(keys[0])        # CMB fluctuations
dust_true = 0.1 * landscape.uniform(keys[1], minval=0, maxval=1)  # Dust template
sync_true = 0.05 * landscape.uniform(keys[2], minval=0, maxval=1) # Synchrotron template

# Stack components: shape (n_components, n_stokes * n_pix)
true_components = jnp.stack([
    cmb_true.flatten(),
    dust_true.flatten(), 
    sync_true.flatten()
])

# Generate observed data: mixing_matrix @ true_components
observed_data = mixing_matrix @ true_components
print(f"Observed data shape: {observed_data.shape}")  # (n_freq, n_stokes * n_pix)
```

### Add Realistic Noise

```python
# Add frequency-dependent noise
# Higher noise at higher frequencies (typical for Planck-like experiment)
noise_levels = jnp.array([2.0, 2.5, 4.5, 2.0, 2.2, 4.8, 14.7])  # μK_CMB RMS

# Generate correlated noise per frequency
noise_key = jax.random.PRNGKey(123)
noise_keys = jax.random.split(noise_key, n_freq)

noise_realizations = []
for i, (noise_level, nkey) in enumerate(zip(noise_levels, noise_keys)):
    # Simple white noise per frequency (could be made more realistic)
    freq_noise = noise_level * landscape.normal(nkey).flatten()
    noise_realizations.append(freq_noise)

noise_data = jnp.stack(noise_realizations)
noisy_observed_data = observed_data + noise_data
```

## Linear Component Separation

### Maximum Likelihood Approach

```python
# Create noise covariance matrix (assuming white noise)
noise_covariance_blocks = []
for noise_level in noise_levels:
    # Noise covariance for this frequency
    noise_var = noise_level**2 * jnp.ones(landscape.size)
    freq_noise_cov = DiagonalOperator(noise_var)
    noise_covariance_blocks.append(freq_noise_cov)

# Block diagonal noise covariance across frequencies
noise_covariance = BlockDiagonalOperator(noise_covariance_blocks)

# Inverse noise covariance (noise weighting)
inv_noise_covariance = BlockDiagonalOperator([
    DiagonalOperator(1.0 / (noise_level**2 * jnp.ones(landscape.size)))
    for noise_level in noise_levels
])
```

### Solve for Components

```python
import lineax as lx
from furax import Config

# Set up the normal equations: (A^T N^-1 A) s = A^T N^-1 d
# where A = mixing_matrix, N^-1 = inv_noise_covariance, d = data, s = components

At_Ninv = mixing_matrix.T @ inv_noise_covariance
At_Ninv_A = At_Ninv @ mixing_matrix
At_Ninv_d = At_Ninv @ noisy_observed_data.flatten()

print(f"Normal equation matrix shape: {At_Ninv_A.shape}")
print(f"Right-hand side shape: {At_Ninv_d.shape}")

# Solve the system
with Config(solver=lx.CG(rtol=1e-8, max_steps=1000)):
    solution = lx.linear_solve(At_Ninv_A, At_Ninv_d)
    
recovered_components = solution.value
print(f"Solver converged: {solution.result}")
print(f"Recovered components shape: {recovered_components.shape}")
```

### Analyze Results

```python
# Reshape recovered components
n_components = 3
recovered_cmb = recovered_components[:landscape.size]
recovered_dust = recovered_components[landscape.size:2*landscape.size]  
recovered_sync = recovered_components[2*landscape.size:]

# Convert back to Stokes format
recovered_cmb_stokes = StokesIQU.from_array(recovered_cmb.reshape(3, -1))
recovered_dust_stokes = StokesIQU.from_array(recovered_dust.reshape(3, -1))
recovered_sync_stokes = StokesIQU.from_array(recovered_sync.reshape(3, -1))

# Compute residuals
cmb_residual = jnp.mean((recovered_cmb - cmb_true.flatten())**2)
dust_residual = jnp.mean((recovered_dust - dust_true.flatten())**2)
sync_residual = jnp.mean((recovered_sync - sync_true.flatten())**2)

print(f"CMB recovery RMS: {jnp.sqrt(cmb_residual):.4f}")
print(f"Dust recovery RMS: {jnp.sqrt(dust_residual):.4f}")
print(f"Synchrotron recovery RMS: {jnp.sqrt(sync_residual):.4f}")
```

## Advanced Component Separation

### Including Priors

```python
from furax.core import SymmetricBandToeplitzOperator

# Add spatial priors (e.g., smoothness prior for CMB)
def create_smoothing_prior(landscape, correlation_length_arcmin=60):
    """Create a smoothness prior operator."""
    # This is a simplified example - real implementation would use 
    # spherical harmonics or proper correlation functions
    
    # Create a band-limited Toeplitz approximation
    n_bands = min(10, landscape.npix // 10)
    bands = []
    for i in range(n_bands):
        if i == 0:
            # Main diagonal
            band = jnp.ones(landscape.npix)
        else:
            # Off-diagonals with exponential decay
            decay = jnp.exp(-i / 5.0)
            band = decay * jnp.ones(landscape.npix - i)
        bands.append(band)
    
    return SymmetricBandToeplitzOperator(bands)

# Create priors for each component
cmb_prior = create_smoothing_prior(landscape)
dust_prior = DiagonalOperator(jnp.ones(landscape.size))  # Flat prior for dust
sync_prior = DiagonalOperator(jnp.ones(landscape.size))  # Flat prior for sync

# Block diagonal prior matrix
prior_matrix = BlockDiagonalOperator([cmb_prior, dust_prior, sync_prior])
```

### Bayesian Solution

```python
# Bayesian solution: (A^T N^-1 A + P^-1) s = A^T N^-1 d
# where P^-1 is the inverse prior covariance

prior_weight = 0.1  # Adjust prior strength
regularized_matrix = At_Ninv_A + prior_weight * prior_matrix

# Solve regularized system
with Config(solver=lx.CG(rtol=1e-8, max_steps=1000)):
    regularized_solution = lx.linear_solve(regularized_matrix, At_Ninv_d)

regularized_components = regularized_solution.value
print(f"Regularized solver converged: {regularized_solution.result}")
```

### Multi-scale Analysis

```python
# Analyze results at different angular scales
def compute_angular_power_spectrum(stokes_map, landscape):
    """Compute angular power spectrum (simplified)."""
    # In practice, would use healpy.sphtfunc.anafast or similar
    # This is a placeholder for demonstration
    intensity = stokes_map.I
    return jnp.var(intensity)  # Simplified variance as proxy for power

# Compare power spectra
true_cmb_power = compute_angular_power_spectrum(cmb_true, landscape)
recovered_cmb_power = compute_angular_power_spectrum(recovered_cmb_stokes, landscape)

print(f"True CMB power: {true_cmb_power:.4f}")
print(f"Recovered CMB power: {recovered_cmb_power:.4f}")
print(f"Power recovery ratio: {recovered_cmb_power/true_cmb_power:.3f}")
```

## Validation and Diagnostics

### Cross-validation

```python
# Split data into training and validation sets
def split_frequency_data(data, train_indices, val_indices):
    """Split multi-frequency data for cross-validation."""
    train_data = data[train_indices]
    val_data = data[val_indices]
    return train_data, val_data

# Use subset of frequencies for training
train_freq_idx = jnp.array([0, 1, 3, 5])  # Skip some frequencies
val_freq_idx = jnp.array([2, 4, 6])       # Validate on held-out frequencies

train_data, val_data = split_frequency_data(
    noisy_observed_data, train_freq_idx, val_freq_idx
)

# Train on subset
train_mixing = MixingMatrixOperator([
    CMBOperator(frequencies[train_freq_idx]),
    DustOperator(frequencies[train_freq_idx], beta=1.54, T_dust=20.0),
    SynchrotronOperator(frequencies[train_freq_idx], beta=-3.1)
])

# Solve using training data only
# ... (similar to above but with train_mixing and train_data)
```

### Residual Analysis

```python
# Analyze fit quality
predicted_data = mixing_matrix @ recovered_components
residuals = noisy_observed_data.flatten() - predicted_data

# Chi-square per frequency
chi2_per_freq = []
for i, noise_level in enumerate(noise_levels):
    freq_residuals = residuals[i*landscape.size:(i+1)*landscape.size]
    chi2 = jnp.sum((freq_residuals / noise_level)**2) / landscape.size
    chi2_per_freq.append(chi2)

print("Chi-square per frequency:")
for i, (freq, chi2) in enumerate(zip(frequencies, chi2_per_freq)):
    print(f"  {freq:6.1f} GHz: χ² = {chi2:.3f}")

overall_chi2 = jnp.mean(jnp.array(chi2_per_freq))
print(f"Overall χ²: {overall_chi2:.3f}")
```

## Visualizing Results

```python
# Visualization utilities (would typically use matplotlib + healpy)
def summarize_component(component_name, true_map, recovered_map):
    """Print summary statistics for a recovered component."""
    correlation = jnp.corrcoef(true_map.flatten(), recovered_map.flatten())[0,1]
    rms_error = jnp.sqrt(jnp.mean((true_map.flatten() - recovered_map.flatten())**2))
    
    print(f"{component_name}:")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  RMS Error: {rms_error:.4f}")
    print(f"  True RMS: {jnp.std(true_map.flatten()):.4f}")
    print(f"  Recovered RMS: {jnp.std(recovered_map.flatten()):.4f}")
    print()

# Summarize all components
summarize_component("CMB", cmb_true.flatten(), recovered_cmb)
summarize_component("Dust", dust_true.flatten(), recovered_dust)  
summarize_component("Synchrotron", sync_true.flatten(), recovered_sync)
```

## Performance Considerations

For large-scale component separation:

1. **Memory Management**: Use block processing for high-resolution maps
2. **Iterative Solvers**: Choose appropriate solver tolerances and preconditioners
3. **Parallelization**: Leverage JAX's `pmap` for multi-GPU processing
4. **Numerical Stability**: Monitor condition numbers of normal equation matrices

```python
# Example: Check condition number
condition_number = jnp.linalg.cond(At_Ninv_A.as_matrix())
print(f"Condition number: {condition_number:.2e}")

if condition_number > 1e12:
    print("Warning: Matrix is poorly conditioned!")
    print("Consider: regularization, better priors, or different frequency selection")
```

This example demonstrates the power of Furax for component separation, showing how linear operators and Stokes data structures work together to solve complex astrophysical problems with clean, composable code.