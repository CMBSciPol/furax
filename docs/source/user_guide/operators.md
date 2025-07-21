# Linear Operators

Linear operators are the computational backbone of Furax, providing composable building blocks for solving inverse problems in CMB analysis. Built on top of JAX and Lineax, Furax operators support mathematical composition, automatic differentiation, and efficient GPU computation.

## Core Concepts

### Abstract Linear Operator

All operators in Furax inherit from `AbstractLinearOperator`, which extends Lineax operators with additional functionality:

- **Composition**: Operators can be composed using the `@` operator (matrix multiplication)
- **Addition**: Operators can be added using the `+` operator
- **Scalar Operations**: Support for scalar multiplication and division
- **Matrix Representation**: Convert to explicit matrices for debugging with `as_matrix()`
- **Properties**: Automatic inference of mathematical properties (symmetric, positive definite, etc.)

```python
from furax.core import DiagonalOperator, BlockDiagonalOperator

# Create operators
op1 = DiagonalOperator(jnp.array([1., 2., 3.]))
op2 = DiagonalOperator(jnp.array([2., 1., 1.]))

# Composition (matrix multiplication)
composed = op1 @ op2

# Addition
summed = op1 + op2

# Scalar operations
scaled = 2.5 * op1
divided = op1 / 3.0

# Check properties
print(f"Is symmetric: {op1.symmetric}")
print(f"Is positive definite: {op1.positive_semidefinite}")
```

## Operator Types

### Diagonal Operators

Perfect for pixel-based weighting, noise covariance, and preconditioning.

**DiagonalOperator**

```python
import jax.numpy as jnp
from furax.core import DiagonalOperator

# Create a diagonal operator for weighting
weights = jnp.array([1.0, 0.5, 2.0, 1.5])
weight_op = DiagonalOperator(weights)

# Apply to data
data = jnp.array([1., 2., 3., 4.])
weighted_data = weight_op @ data
print(weighted_data)  # [1.0, 1.0, 6.0, 6.0]
```

**BroadcastDiagonalOperator**

For operations that need broadcasting across multiple dimensions:

```python
from furax.core import BroadcastDiagonalOperator

# Diagonal values to broadcast
diag_values = jnp.array([1., 2., 3.])  # Shape: (3,)

# Create operator that broadcasts to (3, 4) arrays
broadcast_op = BroadcastDiagonalOperator(
    diagonal=diag_values,
    axis=0  # Broadcast along first axis
)

# Apply to multi-dimensional data
data = jnp.ones((3, 4))
result = broadcast_op @ data
# Each row is scaled by corresponding diagonal value
```

### Block Operators

Essential for multi-component analysis and structured linear systems.

**BlockDiagonalOperator**

```python
from furax.core import BlockDiagonalOperator, DiagonalOperator

# Create individual block operators
block1 = DiagonalOperator(jnp.array([1., 2.]))
block2 = DiagonalOperator(jnp.array([3., 4., 5.]))
block3 = DiagonalOperator(jnp.array([6.]))

# Create block diagonal operator
block_diag = BlockDiagonalOperator([block1, block2, block3])

# Apply to concatenated data
data = jnp.array([1., 1., 1., 1., 1., 1.])  # Length = 2+3+1
result = block_diag @ data
print(result)  # [1., 2., 3., 4., 5., 6.]
```

**BlockRowOperator**

For horizontal concatenation `[A B C]`:

```python
from furax.core import BlockRowOperator

op1 = DiagonalOperator(jnp.array([1., 2.]))
op2 = DiagonalOperator(jnp.array([3., 4.]))

# Create row block: [op1 op2]
row_op = BlockRowOperator([op1, op2])

# Input has combined size
data = jnp.array([1., 1., 1., 1.])  # Size = 2 + 2
result = row_op @ data  # Output size = 2
```

**BlockColumnOperator**

For vertical stacking:

```python
from furax.core import BlockColumnOperator

op1 = DiagonalOperator(jnp.array([1., 2.]))
op2 = DiagonalOperator(jnp.array([3., 4.]))

# Create column block
col_op = BlockColumnOperator([op1, op2])

data = jnp.array([1., 1.])  # Input size = 2
result = col_op @ data      # Output size = 2 + 2 = 4
```

### Toeplitz Operators

Efficient for convolution-like operations and correlated noise modeling.

```python
from furax.core import SymmetricBandToeplitzOperator

# Define the bands for a symmetric Toeplitz matrix
# bands[0] = main diagonal, bands[1] = first off-diagonal, etc.
bands = [
    jnp.array([2., 2., 2., 2.]),      # Main diagonal
    jnp.array([1., 1., 1.]),          # First off-diagonal
    jnp.array([0.5, 0.5])             # Second off-diagonal
]

# Create symmetric band Toeplitz operator
toeplitz_op = SymmetricBandToeplitzOperator(bands)

# Apply to data
data = jnp.array([1., 0., 0., 0.])
result = toeplitz_op @ data
print(result)  # Shows the first column of the Toeplitz matrix
```

### Index and Reshape Operators

For data manipulation and restructuring.

**IndexOperator**

```python
from furax.core import IndexOperator

# Select specific indices
indices = jnp.array([0, 2, 4])
index_op = IndexOperator(indices, input_size=5)

data = jnp.array([10., 20., 30., 40., 50.])
result = index_op @ data
print(result)  # [10., 30., 50.]
```

**ReshapeOperator**

```python
from furax.core import ReshapeOperator

# Reshape from (6,) to (2, 3)
reshape_op = ReshapeOperator(
    input_shape=(6,),
    output_shape=(2, 3)
)

data = jnp.array([1., 2., 3., 4., 5., 6.])
result = reshape_op @ data
print(result.shape)  # (2, 3)
```

**MoveAxisOperator**

```python
from furax.core import MoveAxisOperator

# Move axis from position 0 to position 1
moveaxis_op = MoveAxisOperator(
    source=0, destination=1, shape=(3, 4)
)

data = jnp.ones((3, 4))
result = moveaxis_op @ data
print(result.shape)  # (4, 3)
```

### Tree Operators

For working with PyTree structures (nested dictionaries/lists of arrays):

```python
from furax.core import TreeOperator

# Define operations for each leaf of a PyTree
tree_structure = {
    'I': DiagonalOperator(jnp.array([1., 2.])),
    'Q': DiagonalOperator(jnp.array([3., 4.])),
    'U': DiagonalOperator(jnp.array([5., 6.]))
}

tree_op = TreeOperator(tree_structure)

# Apply to PyTree data
data = {
    'I': jnp.array([1., 1.]),
    'Q': jnp.array([1., 1.]),
    'U': jnp.array([1., 1.])
}

result = tree_op @ data
# Each component is processed by its corresponding operator
```

## Advanced Operator Composition

### Complex Analysis Pipelines

Operators can be composed to create sophisticated analysis pipelines:

```python
from furax.core import (
    DiagonalOperator, BlockDiagonalOperator,
    IndexOperator, ReshapeOperator
)
from furax.obs import HealpixLandscape

# Create a landscape for QU polarization
landscape = HealpixLandscape(nside=8, stokes='QU')

# 1. Noise weighting (inverse variance)
noise_var = jnp.ones(landscape.size)
noise_weighting = DiagonalOperator(1.0 / noise_var)

# 2. Pixel selection (mask bad pixels)
good_pixels = jnp.arange(landscape.size)[::2]  # Select every other pixel
pixel_selection = IndexOperator(good_pixels, landscape.size)

# 3. Component-wise processing
q_size = landscape.npix
u_size = landscape.npix

q_processor = DiagonalOperator(jnp.ones(q_size))
u_processor = DiagonalOperator(2.0 * jnp.ones(u_size))
component_processor = BlockDiagonalOperator([q_processor, u_processor])

# Compose the full pipeline
analysis_pipeline = pixel_selection @ component_processor @ noise_weighting

# Apply to data
data = landscape.normal(jax.random.PRNGKey(0))
processed_data = analysis_pipeline @ data
```

### Iterative Solvers

Furax operators work seamlessly with Lineax solvers:

```python
import lineax as lx
from furax.core import SymmetricBandToeplitzOperator
from furax import Config

# Create a positive definite operator for solving Ax = b
bands = [
    jnp.array([3., 3., 3., 3.]),      # Diagonal dominance ensures PD
    jnp.array([1., 1., 1.])           # Off-diagonal
]
A = SymmetricBandToeplitzOperator(bands)

# Right-hand side
b = jnp.array([1., 2., 3., 4.])

# Solve with conjugate gradient
with Config(solver=lx.CG(rtol=1e-8, max_steps=100)):
    solution = lx.linear_solve(A, b)

print(f"Solution: {solution.value}")
print(f"Converged: {solution.result}")
```

### Matrix-Free Operations

Operators support matrix-free computations for memory efficiency:

```python
def large_scale_analysis(operator, data):
    """Perform analysis without forming explicit matrices."""

    # Matrix-vector product (never forms the full matrix)
    result = operator @ data

    # Operator norms and properties
    print(f"Operator properties:")
    print(f"  Symmetric: {operator.symmetric}")
    print(f"  Positive semidefinite: {operator.positive_semidefinite}")

    return result

# Even for very large operators, memory usage stays manageable
large_diagonal = DiagonalOperator(jnp.ones(1_000_000))
large_data = jnp.ones(1_000_000)

result = large_scale_analysis(large_diagonal, large_data)
```

## Operator Properties

### Mathematical Properties

Furax automatically infers and tracks mathematical properties:

```python
from furax.core import DiagonalOperator, SymmetricBandToeplitzOperator

# Diagonal operators are automatically symmetric and PSD if diagonal > 0
positive_diag = DiagonalOperator(jnp.array([1., 2., 3.]))
print(f"Symmetric: {positive_diag.symmetric}")                    # True
print(f"Positive semidefinite: {positive_diag.positive_semidefinite}")  # True

# Properties are preserved under composition when appropriate
another_positive = DiagonalOperator(jnp.array([2., 1., 4.]))
composed = positive_diag @ another_positive
print(f"Composed is symmetric: {composed.symmetric}")             # True
```

### Custom Operators

You can create custom operators by inheriting from `AbstractLinearOperator`:

```python
from furax.core import AbstractLinearOperator
import jax.numpy as jnp
from jaxtyping import Array, Float

class CustomScalingOperator(AbstractLinearOperator):
    """Custom operator that scales by a factor."""

    def __init__(self, scale_factor: float, size: int):
        self.scale_factor = scale_factor
        self._size = size

    def __call__(self, x: Float[Array, "n"]) -> Float[Array, "n"]:
        return self.scale_factor * x

    @property
    def size(self) -> int:
        return self._size

    @property
    def symmetric(self) -> bool:
        return True  # Scaling is symmetric

    @property
    def positive_semidefinite(self) -> bool:
        return self.scale_factor >= 0

# Use the custom operator
custom_op = CustomScalingOperator(scale_factor=2.5, size=4)
data = jnp.array([1., 2., 3., 4.])
result = custom_op @ data
print(result)  # [2.5, 5.0, 7.5, 10.0]
```

## Performance Considerations

### JAX Transformations

Operators work efficiently with JAX transformations:

```python
# JIT compilation
@jax.jit
def fast_operator_apply(op, data):
    return op @ data

op = DiagonalOperator(jnp.array([1., 2., 3., 4.]))
data = jnp.array([1., 1., 1., 1.])

# First call compiles, subsequent calls are fast
result = fast_operator_apply(op, data)

# Vectorization
@jax.vmap
def batch_apply(data_batch):
    return op @ data_batch

# Apply operator to batch of data
data_batch = jnp.ones((10, 4))  # 10 samples of size 4
results = batch_apply(data_batch)
```

### Memory Efficiency

For large-scale problems:

1. **Use appropriate operator types**: Diagonal operators are more memory-efficient than dense operators
2. **Avoid explicit matrix formation**: Use `operator @ data` instead of `operator.as_matrix() @ data`
3. **Consider block structure**: Block operators can reduce memory usage for structured problems
4. **Use appropriate precision**: Float32 vs Float64 trade-offs

```python
import jax

# Use lower precision for memory efficiency
with jax.config.context(x64_enable=False):
    efficient_op = DiagonalOperator(jnp.ones(1_000_000))
    efficient_data = jnp.ones(1_000_000)
    result = efficient_op @ data  # Uses float32
```

## Best Practices

1. **Compose operators logically**: Build complex operations from simple, well-understood components

2. **Leverage mathematical properties**: Use symmetric, positive definite operators when possible for better solver performance

3. **Test with small examples**: Verify operator behavior with `as_matrix()` on small problems

4. **Profile memory usage**: For large problems, monitor memory consumption

5. **Use appropriate solvers**: Match solver choice to operator properties (e.g., CG for symmetric positive definite systems)

6. **Batch operations**: Use `vmap` to process multiple datasets efficiently
