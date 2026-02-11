# Linear Operators

Linear operators are the computational backbone of Furax, providing composable building blocks for solving inverse problems in CMB analysis. Built on top of JAX, Furax operators support mathematical composition, automatic differentiation, and efficient GPU computation.

## Core Concepts

### Abstract Linear Operator

All operators in Furax inherit from `AbstractLinearOperator`, which extends Lineax operators with additional functionality:

- **Composition**: Operators can be composed using the `@` operator (matrix multiplication)
- **Addition**: Operators can be added using the `+` operator
- **Scalar Operations**: Support for scalar multiplication and division
- **Matrix Representation**: Convert to explicit matrices for debugging with `as_matrix()`
- **Properties**: Automatic inference of mathematical properties (symmetric, positive definite, etc.)

```python
import jax.numpy as jnp

from furax.core import DiagonalOperator

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

# Operators have tags, which are determined statically (not inspecting values)
print(f'Is symmetric: {op1.is_symmetric}')  # True
print(f'Is positive definite: {op1.is_positive_semidefinite}')  # False (!)
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
weighted_data = weight_op(data)
print(weighted_data)  # [1. 1. 6. 6.]
```

**BroadcastDiagonalOperator**

For operations that need broadcasting across multiple dimensions:

```python
import jax
import jax.numpy as jnp

from furax.core import BroadcastDiagonalOperator

# Diagonal values to broadcast
diag_values = jnp.array([1., 2., 3.])

# Create operator that broadcasts to (3, 4) arrays
broadcast_op = BroadcastDiagonalOperator(
    diagonal=diag_values,
    axis_destination=0,  # Broadcast along axes except first one
    in_structure=jax.ShapeDtypeStruct((3, 4), jnp.float32)
)

# Apply to multi-dimensional data
data = jnp.ones((3, 4))
broadcast_op(data)
# Each row is scaled by corresponding diagonal value
# Array([[1., 1., 1., 1.],
#        [2., 2., 2., 2.],
#        [3., 3., 3., 3.]], dtype=float32)
```

### Block Operators

Essential for multi-component analysis and structured linear systems. The blocks can be assembled using
any Python container, as long as it is a PyTree.

**BlockDiagonalOperator**

```python
import jax.numpy as jnp

from furax.core import BlockDiagonalOperator, DiagonalOperator

# Create individual block operators
block1 = DiagonalOperator(jnp.array([1., 2.]))
block2 = DiagonalOperator(jnp.array([3., 4., 5.]))
block3 = DiagonalOperator(jnp.array([6.]))

# Create block diagonal operator using a tuple of operators
block_op = BlockDiagonalOperator((block1, block2, block3))
block_op.as_matrix()
# Array([[1., 0., 0., 0., 0., 0.],
#        [0., 2., 0., 0., 0., 0.],
#        [0., 0., 3., 0., 0., 0.],
#        [0., 0., 0., 4., 0., 0.],
#        [0., 0., 0., 0., 5., 0.],
#        [0., 0., 0., 0., 0., 6.]], dtype=float32)

# Apply to a tuple of data
data = (jnp.array([1., 1.]), jnp.array([1., 1., 1.]), jnp.array([1.]))
result = block_op(data)
print(result)
# (
#     Array([1., 2.], dtype=float32),
#     Array([3., 4., 5.], dtype=float32),
#     Array([6.], dtype=float32),
# )
```

**BlockRowOperator**

For horizontal concatenation `[A B C]`:

```python
import jax
import jax.numpy as jnp

from furax import BlockRowOperator, DenseBlockDiagonalOperator, DiagonalOperator

op1 = DiagonalOperator(jnp.array([1., 2.]))
op2 = DenseBlockDiagonalOperator(jnp.array([[3., 0., 4.], [0., 5., 0.]]), in_structure=jax.ShapeDtypeStruct((3,), jnp.float32))

# Create row block: [op1, op2]
row_op = BlockRowOperator([op1, op2])
row_op.as_matrix()
# Array([[1., 0., 3., 0., 4.],
#        [0., 2., 0., 5., 0.]], dtype=float32)

# Input has combined size
data = [jnp.array([1., 1.]), jnp.array([1., 1., 1.])]
row_op(data)
# Array([8., 7.], dtype=float32)
```

**BlockColumnOperator**

For vertical stacking:

```python
import jax.numpy as jnp

from furax import BlockColumnOperator, DiagonalOperator

op1 = DiagonalOperator(jnp.array([1., 2.]))
op2 = DiagonalOperator(jnp.array([3., 4.]))

# Create column block
col_op = BlockColumnOperator({'x': op1, 'y': op2})
col_op.as_matrix()
# Array([[1., 0.],
#        [0., 2.],
#        [3., 0.],
#        [0., 4.]], dtype=float32)

data = jnp.array([1., 1.])
col_op(data)
# {
#     'x': Array([1., 2.], dtype=float32),
#     'y': Array([3., 4.], dtype=float32),
# }
```

### Toeplitz Operators

Efficient for convolution-like operations and correlated noise modeling.

```python
import jax
import jax.numpy as jnp

from furax import SymmetricBandToeplitzOperator

# Define the bands for a symmetric Toeplitz matrix
bands = jnp.array([[2., 1., 0.5], [1, 0.8, 0.1]])

# Create symmetric band Toeplitz operator
toeplitz_op = SymmetricBandToeplitzOperator(bands, in_structure=jax.ShapeDtypeStruct((2, 6), jnp.float32))
toeplitz_op.as_matrix()
# Array([[2. , 1. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
#        [1. , 2. , 1. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
#        [0.5, 1. , 2. , 1. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
#        [0. , 0.5, 1. , 2. , 1. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. ],
#        [0. , 0. , 0.5, 1. , 2. , 1. , 0. , 0. , 0. , 0. , 0. , 0. ],
#        [0. , 0. , 0. , 0.5, 1. , 2. , 0. , 0. , 0. , 0. , 0. , 0. ],
#        [0. , 0. , 0. , 0. , 0. , 0. , 1. , 0.8, 0.1, 0. , 0. , 0. ],
#        [0. , 0. , 0. , 0. , 0. , 0. , 0.8, 1. , 0.8, 0.1, 0. , 0. ],
#        [0. , 0. , 0. , 0. , 0. , 0. , 0.1, 0.8, 1. , 0.8, 0.1, 0. ],
#        [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.1, 0.8, 1. , 0.8, 0.1],
#        [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.1, 0.8, 1. , 0.8],
#        [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.1, 0.8, 1. ]],      dtype=float32)

# Apply to data
data = jnp.array([1., 0., 0., 0., 0., 0.])
toeplitz_op(data)
# Array([[2. , 1. , 0.5, 0. , 0. , 0. ],
#        [1. , 0.8, 0.1, 0. , 0. , 0. ]], dtype=float32)
```

### Index and Reshape Operators

For data manipulation and restructuring.

**IndexOperator**

```python
import jax
import jax.numpy as jnp

from furax import IndexOperator

# Select specific indices
indices = jnp.array([0, 2, 4])
index_op = IndexOperator(indices, in_structure=jax.ShapeDtypeStruct((5,), jnp.float32))

data = jnp.array([10., 20., 30., 40., 50.])
index_op(data)
# Array([10., 30., 50.], dtype=float32)
```

**ReshapeOperator**

```python
import jax
import jax.numpy as jnp

from furax import ReshapeOperator

# Reshape from (6,) to (2, 3)
reshape_op = ReshapeOperator(
    shape=(2, 3),
    in_structure={'x': jax.ShapeDtypeStruct((6,), jnp.float32)},
)

data = {'x': jnp.array([1., 2., 3., 4., 5., 6.])}
reshape_op(data)
# {
#     'x': Array([[1., 2., 3.],
#                 [4., 5., 6.]], dtype=float32)
# }
```

**MoveAxisOperator**

```python
import jax
import jax.numpy as jnp

from furax import MoveAxisOperator

# Move axis from position 0 to position 1
moveaxis_op = MoveAxisOperator(
    source=0, destination=1, in_structure=[jax.ShapeDtypeStruct((3, 4), jnp.float32)]
)

data = [jnp.arange(12).reshape((6, 2))]
moveaxis_op(data)
# [Array([[ 0,  2,  4,  6,  8, 10],
#         [ 1,  3,  5,  7,  9, 11]], dtype=int32)]

```

### Tree Operators

For working with PyTree structures (nested dictionaries/lists of arrays):

```python
from furax import DiagonalOperator, TreeOperator

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
from furax import (
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

Furax operators work seamlessly with jnp.scipy and lineax solvers:

```python
import jax
import jax.numpy as jnp

import lineax as lx
from furax import SymmetricBandToeplitzOperator

# Create a positive definite operator for solving Ax = b
band = jnp.array([1., 0.5, 0.25, 0.125])
A = SymmetricBandToeplitzOperator(band, in_structure=jax.ShapeDtypeStruct((6,), jnp.float32), method='direct')

# Right-hand side
b = jnp.array([1., 2., 3., 4., 5., 6.])

# Solve with conjugate gradient
solver=lx.CG(atol=1e-5, rtol=1e-5, max_steps=100)
solution = A.I(solver=solver)(b)
print(f'CG solution: {solution}')
# CG solution: [0.13638389 0.9942686  0.802103   1.3276514  1.6366035  4.7495227 ]

# Solve with GMRES
solution = jax.scipy.sparse.linalg.gmres(A, b)[0]
print(f'GMRES solution: {solution}')
# GMRES solution: [0.13638386 0.9942684  0.8021034  1.3276513  1.6366041  4.749522  ]

expected_solution = jnp.linalg.inv(A.as_matrix()) @ b
print(f"Expected solution: {expected_solution}")
# Expected solution: [0.13638361 0.99426854 0.802103   1.3276504  1.636604   4.749522  ]
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

Furax statically stores algebraic properties of operators (such as squareness, symmetry, orthogonality, ...). Note that these properties are not inferred from the operator's
data since it would not work after jitting the operation. These properties are not yet propagated during composition
(but note that most properties such as symmetry, positive semi-definiteness are not preserved in general)

```python
import jax.numpy as jnp

from furax import DiagonalOperator

# Diagonal operators are automatically symmetric and PSD if diagonal > 0
positive_diag = DiagonalOperator(jnp.array([1., 2., 3.]))
print(f"Square: {positive_diag.is_square}")  # True
print(f"Symmetric: {positive_diag.is_symmetric}")  # True
print(f"Positive semidefinite: {positive_diag.is_positive_semidefinite}")  # False
```

### Custom Operators

You can create custom operators by inheriting from `AbstractLinearOperator`:

```python
from dataclasses import field

import jax
from jax import Array
import jax.numpy as jnp
from jaxtyping import PyTree, Inexact

from furax import AbstractLinearOperator, symmetric

@symmetric
class CustomScalingOperator(AbstractLinearOperator):
    """Custom operator that scales a PyTree by a static factor."""

    scale_factor: float = field(metadata={'static': True})

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        return jax.tree.map(lambda leave: self.scale_factor * leave, x)


# Use the custom operator
custom_op = CustomScalingOperator(scale_factor=2.5, in_structure={'input': jax.ShapeDtypeStruct((3,), jnp.float32)})
data = {'input': jnp.array([1., 2., 3.])}
custom_op(data)
# {
#     'input': Array([ 2.5,  5. ,  7.5 ], dtype=float32)
# }

# The operator is symmetric
print(custom_op.is_symmetric)  # True
assert custom_op.T is custom_op

# Square operators can be inverted (default: CG)
custom_op.I(data)
# {
#     'input': Array([0.4, 0.8, 1.2], dtype=float32)
# }
```

## Performance Considerations

### JAX Transformations

Operators work efficiently with JAX transformations. Since operators are PyTrees, they can be jitted, vmapped etc.

```python
import jax
import jax.numpy as jnp

from furax import DiagonalOperator

# JIT compilation
@jax.jit
def fast_operator_apply(op, data):
    return op(data)

op = DiagonalOperator(jnp.array([1., 2., 3., 4.]))
data = jnp.array([1., 1., 1., 1.])
op(data)
# Array([1., 2., 3., 4.], dtype=float32)

# First call compiles, subsequent calls are fast
fast_operator_apply(op, data)
# Array([1., 2., 3., 4.], dtype=float32)

# Vectorization
@jax.vmap
def batch_apply(data_batch):
    return op(data_batch)

# Apply operator to batch of data
data_batch = jnp.arange(40).reshape(10, 4)  # 10 samples of size 4
batch_apply(data_batch)
# Array([[  0.,   2.,   6.,  12.],
#        [  4.,  10.,  18.,  28.],
#        [  8.,  18.,  30.,  44.],
#        [ 12.,  26.,  42.,  60.],
#        [ 16.,  34.,  54.,  76.],
#        [ 20.,  42.,  66.,  92.],
#        [ 24.,  50.,  78., 108.],
#        [ 28.,  58.,  90., 124.],
#        [ 32.,  66., 102., 140.],
#        [ 36.,  74., 114., 156.]], dtype=float32)
```

### Memory Efficiency

For large-scale problems:

1. **Use appropriate operator types**: Diagonal operators are more memory-efficient than dense operators
2. **Avoid explicit matrix formation**: Use `operator @ data` instead of `operator.as_matrix() @ data`
3. **Consider block structure**: Block operators can reduce memory usage for structured problems
4. **Use appropriate precision**: Float32 vs Float64 trade-offs


## Best Practices

1. **Compose operators logically**: Build complex operations from simple, well-understood components

2. **Leverage mathematical properties**: Use symmetric, positive definite operators when possible for better solver performance

3. **Test with small examples**: Verify operator behavior with `as_matrix()` on small problems

4. **Profile memory usage**: For large problems, monitor memory consumption

5. **Use appropriate solvers**: Match solver choice to operator properties (e.g., CG for symmetric positive definite systems)

6. **Batch operations**: Use `vmap` to process multiple datasets efficiently
