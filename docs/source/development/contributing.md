# Contributing to Furax

We welcome contributions to Furax! This guide will help you get started with contributing code, documentation, or bug reports.

## Development Setup

### Fork and Clone

1. Fork the Furax repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/your-username/furax.git
cd furax
```

### Development Installation

Install Furax in development mode with all dependencies:

```bash
pip install -e .[dev]
```

This installs:
- Core dependencies (JAX, Lineax, etc.)
- Development tools (pytest, mypy, ruff, pre-commit)
- Documentation tools (sphinx, etc.)

### Pre-commit Hooks

Set up pre-commit hooks to ensure code quality:

```bash
pre-commit install
```

This will automatically run linting, formatting, and type checking on every commit.

## Code Quality Standards

### Formatting and Linting

Furax uses Ruff for both linting and formatting:

```bash
# Check formatting and style
ruff check src/

# Auto-format code
ruff format src/

# Fix auto-fixable issues
ruff check --fix src/
```

Configuration:
- Line length: 100 characters
- String quotes: Single quotes preferred
- Import sorting: Automatic

### Type Checking

We use MyPy for static type checking:

```bash
# Type check the core package
mypy src/furax/
```

Type checking is enforced only on the `src/furax/` directory. External dependencies like `healpy` and `jax-healpy` are ignored.

Key requirements:
- All public functions should have type annotations
- Use `jaxtyping` for array type annotations
- Complex types should be documented

## Testing

### Test Structure

Tests are organized in the `tests/` directory:

```
tests/
├── conftest.py              # Global fixtures
├── core/                    # Linear operator tests
├── obs/                     # Observation framework tests
├── interfaces/              # External interface tests
└── data/                    # Test data files
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage and colored output
pytest -s -ra --color=yes

# Run specific test file
pytest tests/core/test_diagonal.py

# Run tests matching pattern
pytest -k "test_diagonal"

# Run tests with specific markers
pytest -m "slow"
```

### Test Features

- **JAX x64 precision**: Automatically enabled for numerical accuracy
- **Parametrized fixtures**: Tests run with different Stokes combinations (I, QU, IQU, IQUV)
- **Data fixtures**: Cached test data with automatic downloads
- **Custom assertions**: Specialized checks for Furax data types

### UPDATE ME. Writing Tests

Use parametrized fixtures for comprehensive testing:

```python
import pytest
from furax.obs import Stokes

@pytest.mark.parametrize("stokes_fixture", ["I", "QU", "IQU"], indirect=True)
def test_stokes_arithmetic(stokes_fixture):
    stokes_data = stokes_fixture

    # Test addition
    result = stokes_data + stokes_data
    assert isinstance(result, type(stokes_data))

    # Test scalar multiplication
    scaled = 2.0 * stokes_data
    assert isinstance(scaled, type(stokes_data))
```

### HPC Testing

For GPU testing on HPC clusters:

```bash
# Submit to SLURM queue (JeanZay example)
sbatch slurms/astro-sim-v100-testing.slurm
```

## Code Architecture

### Core Principles

1. **Composability**: Linear operators should compose naturally with `@` and `+`
2. **JAX Integration**: All data structures are PyTrees compatible with JAX transformations
3. **Type Safety**: Extensive use of type hints and jaxtyping
4. **Mathematical Clarity**: Code should reflect mathematical operations clearly

### Operator Development

When creating new operators, inherit from `AbstractLinearOperator`:

```python
from dataclasses import field

from furax import AbstractLinearOperator, symmetric
from jaxtyping import Array, Float

@symmetric
class MyCustomOperator(AbstractLinearOperator):
    """A custom homothety operator with static scaling factor.

    Example:

        >>> op = MyCustomOperator(10, in_structure=jax.ShapeDtypeStruct((2,), jnp.float32))
        >>> op(jnp.array([1., 2]))
        Array([10., 20.], dtype=float32)
        >>> op.I(jnp.array([1., 2]))
        Array([0.1, 0.2], dtype=float32)

    """
    scaling_factor: float = field(metadata={'static': True})

    def mv(self, x: Float[Array, "n"]) -> Float[Array, "n"]:
        # Implement the linear operation
        return self.scaling_factor * x

    def inverse(self) -> AbstractLinearOperator:
        # Overrides the default implementation
        return MyCustomOperator(1/self.scaling_factor, in_structure=self.in_structure)

    @property
    def is_negative_semidefinite(self) -> bool:
        return self.scaling_factor <= 0

    @property
    def is_positive_semidefinite(self) -> bool:
        return self.scaling_factor >= 0
```

Key requirements:
- Implement `mv` for matrix-vector multiplication
- Specify mathematical properties when known
- Include comprehensive docstrings with examples

### Data Structure Development

#UPDATE ME. New Stokes classes should follow the established pattern:

```python
from furax.obs import Stokes
from jaxtyping import Array, Float

class StokesXY(Stokes):
    """Custom Stokes parameters for X and Y polarization."""

    X: Float[Array, "n_pix"]
    Y: Float[Array, "n_pix"]

    @classmethod
    def from_stokes(cls, x: Array, y: Array) -> "StokesXY":
        return cls(X=jnp.asarray(x), Y=jnp.asarray(y))

    @property
    def stokes(self) -> str:
        return "XY"
```

## Documentation

### Docstring Style

Use Google-style docstrings with type information:

```python
def my_function(
    data: Float[Array, "n_pix"],
    scale: float = 1.0
) -> Float[Array, "n_pix"]:
    """Process CMB data with scaling.

    Args:
        data: Input CMB map with shape (n_pix,)
        scale: Scaling factor to apply

    Returns:
        Scaled CMB map with same shape as input

    Example:
        >>> import jax.numpy as jnp
        >>> data = jnp.array([1., 2., 3.])
        >>> result = my_function(data, scale=2.0)
        >>> print(result)
        [2. 4. 6.]
    """
    return scale * data
```

### Building Documentation

```bash
# Build HTML documentation
cd docs
make html

# View in browser
open build/html/index.html
```

### Mathematical Notation

Use proper LaTeX for mathematical expressions:

```rst
The maximum likelihood estimator is:

.. math::

   \\hat{m} = (P^T N^{-1} P)^{-1} P^T N^{-1} d

where :math:`P` is the pointing matrix.
```

## Pull Request Process

### Before Submitting

1. Ensure all tests pass: `pytest`
2. Check code quality: `ruff check src/` and `mypy src/furax/`
3. Update documentation if needed
4. Add tests for new functionality

### Pull Request Guidelines

1. **Clear Description**: Explain what the PR does and why
2. **Small, Focused Changes**: One feature or fix per PR
3. **Test Coverage**: Include tests for new code
4. **Documentation**: Update docs for user-facing changes
5. **Backwards Compatibility**: Avoid breaking existing APIs without discussion

Example PR Description:

```
## Summary

Adds support for non-uniform noise in ToeplitzOperator

## Changes

- Modified SymmetricBandToeplitzOperator to accept per-pixel noise scaling
- Added unit tests for new functionality
- Updated documentation with usage examples

## Testing

- All existing tests pass
- New tests added in test_toeplitz.py
- Verified with realistic CMB noise simulation
```

### Review Process

1. Automated checks run on all PRs (tests, linting, type checking)
2. Code review by maintainers
3. Address feedback and update PR
4. Merge once approved and all checks pass

## Issue Reporting

### Bug Reports

Include:
- Clear description of the problem
- Minimal code example that reproduces the issue
- System information (OS, Python version, JAX version)
- Expected vs actual behavior

### Feature Requests

Include:
- Clear description of the desired feature
- Use case or motivation
- Proposed API (if applicable)
- Willingness to implement

## Getting Help

- **GitHub Discussions**: For questions about usage
- **GitHub Issues**: For bug reports and feature requests
- **Code Review**: For feedback on implementation approaches

Thank you for contributing to Furax!
