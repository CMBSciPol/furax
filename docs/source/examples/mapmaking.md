# CMB Mapmaking

This example demonstrates how to use Furax for CMB mapmaking, the process of converting time-ordered data (TOD) from CMB experiments into sky maps. Mapmaking is a fundamental step in CMB data analysis that projects detector measurements onto the sky.

## Overview

CMB mapmaking involves:

1. **Pointing Model**: Converting detector measurements to sky coordinates
2. **Projection Operator**: Mapping between time-ordered data and sky pixels  
3. **Noise Model**: Accounting for detector noise correlations
4. **Linear System**: Solving for maximum likelihood sky maps

Furax provides the linear algebra framework to formulate and solve these problems efficiently.

## Basic Mapmaking

### Setting Up the Sky and Observations

```python
import jax
import jax.numpy as jnp
from furax.obs import HealpixLandscape, StokesIQU
from furax.core import DiagonalOperator, BlockDiagonalOperator
import lineax as lx
from furax import Config

# Define sky resolution
nside = 64
landscape = HealpixLandscape(nside=nside, stokes='IQU')
n_pix = landscape.npix

print(f"Sky map: {n_pix} pixels with {landscape.stokes} Stokes parameters")
print(f"Total sky parameters: {landscape.size}")
```

### Create Simulated Time-Ordered Data

```python
# Simulate scanning strategy
n_samples = 100000  # Number of time samples

key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 4)

# Generate random pointing (in practice, this comes from satellite attitude)
# pixel_indices: which sky pixel each sample observes
pixel_indices = jax.random.randint(
    keys[0], (n_samples,), 0, n_pix
)

# Polarization angles (detector orientation relative to sky)
psi_angles = jax.random.uniform(keys[1], (n_samples,), 0, 2*jnp.pi)

print(f"Time-ordered data: {n_samples} samples")
print(f"Unique pixels observed: {len(jnp.unique(pixel_indices))}")
```

### Create True Sky Signal

```python
# Generate a realistic CMB sky
true_sky = landscape.normal(keys[2])

# Add some large-scale structure (simplified)
large_scale_component = 0.1 * landscape.uniform(
    keys[3], minval=-1, maxval=1
)
true_sky = true_sky + large_scale_component

print(f"True sky shape: {true_sky.shape}")
print(f"Sky RMS (I): {jnp.std(true_sky.I):.3f} μK")
print(f"Sky RMS (Q): {jnp.std(true_sky.Q):.3f} μK") 
print(f"Sky RMS (U): {jnp.std(true_sky.U):.3f} μK")
```

### Build the Pointing Operator

```python
from furax.obs import ProjectionOperator

def create_pointing_matrix(pixel_indices, psi_angles, landscape):
    """Create the pointing matrix that maps sky to TOD."""
    
    n_samples = len(pixel_indices)
    n_sky_params = landscape.size
    
    # For each time sample, create the pointing vector
    pointing_vectors = []
    
    for i in range(n_samples):
        pix_idx = pixel_indices[i]
        psi = psi_angles[i]
        
        # Create pointing vector for this sample
        # For IQU: [I_weight, Q_weight, U_weight] for each pixel
        pointing_vector = jnp.zeros(n_sky_params)
        
        # Intensity always couples with weight 1
        pointing_vector = pointing_vector.at[pix_idx].set(1.0)
        
        # Q couples with cos(2*psi), U couples with sin(2*psi)
        if landscape.stokes in ['QU', 'IQU']:
            q_idx = pix_idx + n_pix  # Q parameters start after I
            u_idx = pix_idx + 2*n_pix  # U parameters start after Q
            
            pointing_vector = pointing_vector.at[q_idx].set(jnp.cos(2*psi))
            pointing_vector = pointing_vector.at[u_idx].set(jnp.sin(2*psi))
        
        pointing_vectors.append(pointing_vector)
    
    # Stack into pointing matrix: (n_samples, n_sky_params)
    return jnp.stack(pointing_vectors)

# Create the pointing matrix
pointing_matrix = create_pointing_matrix(pixel_indices, psi_angles, landscape)
print(f"Pointing matrix shape: {pointing_matrix.shape}")
```

### Simulate Observations

```python
# Convert sky to time-ordered data using pointing matrix
true_tod = pointing_matrix @ true_sky.flatten()

# Add detector noise
noise_level = 10.0  # μK per sample
noise_key = jax.random.PRNGKey(789)
noise_tod = noise_level * jax.random.normal(noise_key, (n_samples,))

# Observed time-ordered data
observed_tod = true_tod + noise_tod

print(f"TOD RMS (signal): {jnp.std(true_tod):.3f} μK")
print(f"TOD RMS (noise): {jnp.std(noise_tod):.3f} μK") 
print(f"TOD RMS (total): {jnp.std(observed_tod):.3f} μK")
print(f"Signal-to-noise ratio: {jnp.std(true_tod)/jnp.std(noise_tod):.2f}")
```

## Maximum Likelihood Mapmaking

### Set Up Linear System

The maximum likelihood estimator for the sky map is:

$$
\hat{m} = (P^T N^{-1} P)^{-1} P^T N^{-1} d
$$

where $P$ is the pointing matrix, $N^{-1}$ is the inverse noise covariance, and $d$ is the data.

```python
from furax.core import DenseOperator

# Create pointing operator from matrix
pointing_op = DenseOperator(pointing_matrix)

# Create noise covariance (assume white noise)
noise_variance = noise_level**2 * jnp.ones(n_samples)
inv_noise_cov = DiagonalOperator(1.0 / noise_variance)

# Build normal equations: P^T N^-1 P
PtNinv = pointing_op.T @ inv_noise_cov
normal_matrix = PtNinv @ pointing_op

# Right-hand side: P^T N^-1 d
rhs = PtNinv @ observed_tod

print(f"Normal matrix shape: {normal_matrix.shape}")
print(f"Right-hand side shape: {rhs.shape}")
```

### Solve for Sky Map

```python
# Solve the linear system
with Config(solver=lx.CG(rtol=1e-8, max_steps=5000)):
    solution = lx.linear_solve(normal_matrix, rhs)

recovered_sky_flat = solution.value
print(f"Solver converged: {solution.result}")
print(f"Iterations: {solution.stats.get('niter', 'N/A')}")

# Reshape back to Stokes format
recovered_sky = StokesIQU.from_array(
    recovered_sky_flat.reshape(3, n_pix)
)

print(f"Recovered sky shape: {recovered_sky.shape}")
```

### Analyze Mapmaking Results

```python
# Compute pixel-wise uncertainties (diagonal of covariance matrix)
# For large problems, approximate using diagonal of normal matrix inverse
normal_diag = jnp.diag(normal_matrix.as_matrix())  # Only for small examples!
pixel_uncertainties = 1.0 / jnp.sqrt(normal_diag)

# Reshape uncertainties
uncertainty_sky = StokesIQU.from_array(
    pixel_uncertainties.reshape(3, n_pix)
)

# Compute residuals
residual_I = recovered_sky.I - true_sky.I
residual_Q = recovered_sky.Q - true_sky.Q  
residual_U = recovered_sky.U - true_sky.U

print(f"Residual RMS (I): {jnp.std(residual_I):.3f} μK")
print(f"Residual RMS (Q): {jnp.std(residual_Q):.3f} μK")
print(f"Residual RMS (U): {jnp.std(residual_U):.3f} μK")

print(f"Average uncertainty (I): {jnp.mean(uncertainty_sky.I):.3f} μK")
print(f"Average uncertainty (Q): {jnp.mean(uncertainty_sky.Q):.3f} μK") 
print(f"Average uncertainty (U): {jnp.mean(uncertainty_sky.U):.3f} μK")
```

## Advanced Mapmaking

### Including Hit Count Weighting

Real experiments have non-uniform sky coverage:

```python
# Compute hit count per pixel (how many times each pixel is observed)
hit_counts = jnp.zeros(n_pix)
for pix_idx in pixel_indices:
    hit_counts = hit_counts.at[pix_idx].add(1.0)

print(f"Average hits per pixel: {jnp.mean(hit_counts):.1f}")
print(f"Min hits: {jnp.min(hit_counts)}, Max hits: {jnp.max(hit_counts)}")

# Pixels with no hits cannot be constrained
observed_pixels = hit_counts > 0
n_observed = jnp.sum(observed_pixels)
print(f"Observed pixels: {n_observed} / {n_pix} ({100*n_observed/n_pix:.1f}%)")
```

### Correlated Noise

Handle temporal correlations in the noise:

```python
from furax.core import SymmetricBandToeplitzOperator

def create_correlated_noise_operator(n_samples, correlation_time=10):
    """Create operator for temporally correlated noise."""
    
    # Create banded Toeplitz matrix for 1/f noise correlation
    max_bands = min(correlation_time, 20)  # Limit for computational efficiency
    bands = []
    
    for i in range(max_bands):
        if i == 0:
            # Main diagonal
            band = jnp.ones(n_samples)
        else:
            # Off-diagonal bands with exponential decay
            decay = jnp.exp(-i / correlation_time)
            band = decay * jnp.ones(n_samples - i)
        bands.append(band)
    
    return SymmetricBandToeplitzOperator(bands)

# Create correlated noise model
corr_noise_cov = create_correlated_noise_operator(n_samples)

# For mapmaking, we need the inverse (expensive for large problems!)
# In practice, would use approximate methods or preconditioning
print(f"Correlated noise covariance shape: {corr_noise_cov.shape}")
```

### Iterative Mapmaking with Preconditioning

```python
# Create a simple preconditioner based on hit counts
def create_hit_count_preconditioner(hit_counts, landscape):
    """Create preconditioner from hit counts."""
    
    # Preconditioner diagonal: higher hits = easier to solve
    precond_diag = jnp.zeros(landscape.size)
    
    # I, Q, U components get the same hit count weighting
    for i in range(3):  # IQU
        start_idx = i * n_pix
        end_idx = (i + 1) * n_pix
        precond_diag = precond_diag.at[start_idx:end_idx].set(
            jnp.sqrt(hit_counts + 1e-10)  # Avoid division by zero
        )
    
    return DiagonalOperator(precond_diag)

preconditioner = create_hit_count_preconditioner(hit_counts, landscape)

# Solve with preconditioning
with Config(solver=lx.CG(rtol=1e-6, max_steps=3000)):
    # Apply preconditioning: solve P^-1 (AtNA) P^-1 (P x) = P^-1 (AtN d)
    preconditioned_matrix = preconditioner @ normal_matrix @ preconditioner
    preconditioned_rhs = preconditioner @ rhs
    
    preconditioned_solution = lx.linear_solve(
        preconditioned_matrix, preconditioned_rhs
    )
    
    # Transform back
    final_solution = preconditioner @ preconditioned_solution.value

print(f"Preconditioned solver converged: {preconditioned_solution.result}")
```

### Multi-Detector Mapmaking

For experiments with multiple detectors:

```python
# Simulate multiple detectors
n_detectors = 4
detector_names = [f"Det_{i:02d}" for i in range(n_detectors)]

# Different noise levels per detector
detector_noise_levels = jnp.array([8.0, 10.0, 12.0, 9.0])  # μK

# Generate TOD for each detector
multi_detector_tod = {}
multi_detector_pointing = {}

for i, det_name in enumerate(detector_names):
    # Each detector has slightly different pointing due to focal plane layout
    det_key = jax.random.PRNGKey(1000 + i)
    det_keys = jax.random.split(det_key, 3)
    
    # Add small random offset to pixel indices (focal plane effects)
    det_pixel_indices = pixel_indices  # Simplified - same pointing
    det_psi_angles = psi_angles + 0.1 * jax.random.normal(det_keys[0], (n_samples,))
    
    # Create pointing matrix for this detector
    det_pointing = create_pointing_matrix(det_pixel_indices, det_psi_angles, landscape)
    multi_detector_pointing[det_name] = det_pointing
    
    # Generate TOD
    det_signal = det_pointing @ true_sky.flatten()
    det_noise = detector_noise_levels[i] * jax.random.normal(det_keys[1], (n_samples,))
    multi_detector_tod[det_name] = det_signal + det_noise
    
    print(f"{det_name}: noise level = {detector_noise_levels[i]:.1f} μK")
```

### Combined Multi-Detector Solution

```python
# Stack all detector data
all_tod = jnp.concatenate([multi_detector_tod[det] for det in detector_names])
all_pointing_matrices = [multi_detector_pointing[det] for det in detector_names]
combined_pointing = jnp.concatenate(all_pointing_matrices, axis=0)

# Create block diagonal noise covariance
noise_cov_blocks = []
for noise_level in detector_noise_levels:
    det_noise_cov = DiagonalOperator(
        (1.0 / noise_level**2) * jnp.ones(n_samples)
    )
    noise_cov_blocks.append(det_noise_cov)

combined_inv_noise_cov = BlockDiagonalOperator(noise_cov_blocks)

# Solve combined system
combined_pointing_op = DenseOperator(combined_pointing)
combined_PtNinv = combined_pointing_op.T @ combined_inv_noise_cov
combined_normal = combined_PtNinv @ combined_pointing_op
combined_rhs = combined_PtNinv @ all_tod

print(f"Combined system shape: {combined_normal.shape}")
print(f"Total samples: {len(all_tod)}")

# Solve
with Config(solver=lx.CG(rtol=1e-6, max_steps=2000)):
    combined_solution = lx.linear_solve(combined_normal, combined_rhs)

multi_det_sky = StokesIQU.from_array(
    combined_solution.value.reshape(3, n_pix)
)

print(f"Multi-detector solution converged: {combined_solution.result}")
```

## Cross-Validation and Diagnostics

### Split-Half Tests

```python
# Split data in half for cross-validation
n_half = n_samples // 2

# First half
pointing_1 = pointing_matrix[:n_half]
tod_1 = observed_tod[:n_half]

# Second half  
pointing_2 = pointing_matrix[n_half:2*n_half]
tod_2 = observed_tod[n_half:2*n_half]

def solve_split_map(pointing_split, tod_split, noise_level):
    """Solve mapmaking for data split."""
    pointing_op = DenseOperator(pointing_split)
    inv_noise = DiagonalOperator(
        (1.0 / noise_level**2) * jnp.ones(len(tod_split))
    )
    
    PtNinv = pointing_op.T @ inv_noise
    normal = PtNinv @ pointing_op
    rhs = PtNinv @ tod_split
    
    with Config(solver=lx.CG(rtol=1e-6, max_steps=2000)):
        solution = lx.linear_solve(normal, rhs)
    
    return StokesIQU.from_array(solution.value.reshape(3, n_pix))

# Solve for both halves
sky_split_1 = solve_split_map(pointing_1, tod_1, noise_level)
sky_split_2 = solve_split_map(pointing_2, tod_2, noise_level) 

# Compare splits (should be consistent within noise)
diff_I = sky_split_1.I - sky_split_2.I
diff_Q = sky_split_1.Q - sky_split_2.Q
diff_U = sky_split_1.U - sky_split_2.U

print(f"Split difference RMS (I): {jnp.std(diff_I):.3f} μK")
print(f"Split difference RMS (Q): {jnp.std(diff_Q):.3f} μK")
print(f"Split difference RMS (U): {jnp.std(diff_U):.3f} μK")
```

### Null Tests

```python
# Create jackknife test: flip sign of alternate samples
jackknife_tod = observed_tod.copy()
flip_indices = jnp.arange(0, n_samples, 2)  # Every other sample
jackknife_tod = jackknife_tod.at[flip_indices].multiply(-1)

# Solve jackknife map (should be consistent with noise)
jackknife_rhs = PtNinv @ jackknife_tod

with Config(solver=lx.CG(rtol=1e-6, max_steps=2000)):
    jackknife_solution = lx.linear_solve(normal_matrix, jackknife_rhs)

jackknife_sky = StokesIQU.from_array(
    jackknife_solution.value.reshape(3, n_pix)
)

print(f"Jackknife map RMS (I): {jnp.std(jackknife_sky.I):.3f} μK")
print(f"Expected from noise: ~{noise_level/jnp.sqrt(jnp.mean(hit_counts)):.3f} μK")
```

## Performance Optimization

### Memory-Efficient Implementation

For very large problems:

```python
def memory_efficient_mapmaking(pixel_indices, psi_angles, tod, noise_level):
    """Memory-efficient mapmaking without storing full pointing matrix."""
    
    # Build normal matrix and RHS without storing pointing matrix
    n_sky_params = landscape.size
    normal_matrix_data = jnp.zeros((n_sky_params, n_sky_params))
    rhs_data = jnp.zeros(n_sky_params)
    
    # Accumulate contributions sample by sample
    inv_noise_var = 1.0 / noise_level**2
    
    for i in range(len(tod)):
        pix_idx = pixel_indices[i]
        psi = psi_angles[i]
        data_val = tod[i]
        
        # Create pointing vector for this sample
        pointing_vec = jnp.zeros(n_sky_params)
        pointing_vec = pointing_vec.at[pix_idx].set(1.0)  # I
        
        if landscape.stokes in ['QU', 'IQU']:
            q_idx = pix_idx + n_pix
            u_idx = pix_idx + 2*n_pix
            pointing_vec = pointing_vec.at[q_idx].set(jnp.cos(2*psi))  # Q
            pointing_vec = pointing_vec.at[u_idx].set(jnp.sin(2*psi))  # U
        
        # Accumulate: P^T N^-1 P and P^T N^-1 d
        normal_matrix_data += inv_noise_var * jnp.outer(pointing_vec, pointing_vec)
        rhs_data += inv_noise_var * data_val * pointing_vec
    
    return DenseOperator(normal_matrix_data), rhs_data

# For demonstration with small subset
subset_size = 1000
small_indices = pixel_indices[:subset_size]
small_psi = psi_angles[:subset_size] 
small_tod = observed_tod[:subset_size]

efficient_normal, efficient_rhs = memory_efficient_mapmaking(
    small_indices, small_psi, small_tod, noise_level
)

print("Memory-efficient method completed")
```

This example demonstrates the complete mapmaking pipeline in Furax, from simulation to advanced analysis techniques. The linear operator framework makes it easy to experiment with different noise models, preconditioners, and multi-detector configurations.