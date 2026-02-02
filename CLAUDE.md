# CLAUDE.md

Furax provides composable building blocks for solving inverse problems in astrophysical and cosmological domains. It is built on JAX and uses linear operators as the core abstraction for modeling observation pipelines and solving map-making problems.

## Architecture

### Package Structure

```
src/furax/
├── core/           # Linear operator framework
│   ├── _base.py    # AbstractLinearOperator and composition/addition
│   ├── _blocks.py  # Block operators (row, column, diagonal)
│   ├── _diagonal.py, _mask.py, _indices.py  # Element-wise operators
│   └── rules.py    # Algebraic simplification rules
├── obs/            # Observation modeling
│   ├── stokes.py   # Stokes vector types (StokesI, StokesQU, StokesIQU)
│   ├── landscapes.py  # Sky map representations (Healpix, WCS)
│   └── operators/  # HWP, polarizer, QU rotation, SEDs
├── mapmaking/      # Map-making algorithms
│   ├── mapmaker.py # Main map-maker implementation
│   ├── noise.py    # Noise models (white, atmospheric)
│   └── config.py   # Configuration classes
├── linalg/         # Eigenvalue solvers (LOBPCG, Lanczos)
├── interfaces/     # External library integrations
│   ├── toast/      # TOAST integration
│   ├── sotodlib/   # sotodlib integration
│   └── litebird_sim/  # LiteBIRD simulation integration
├── math/           # Math utilities (quaternions)
├── preprocessing/  # Data preprocessing (gap filling)
├── io/             # I/O utilities
└── tree.py         # PyTree utilities (dot, norm, add, etc.)
```

### Core Abstraction: Linear Operators

The codebase centers on `AbstractLinearOperator` (in `core/_base.py`), extending `lineax.AbstractLinearOperator`. Operators are Equinox modules (immutable JAX-compatible dataclasses).

**Key operations:**
- `A @ B` - composition (B applied first, then A)
- `A + B` - addition
- `A.T` - transpose (lazy, uses JAX autodiff)
- `A.I` - inverse (lazy, solved via lineax)
- `A(x)` or `A.mv(x)` - apply to input

**Operator tags** (decorators in `core/_base.py`):
- `@diagonal`, `@symmetric`, `@orthogonal` - register lineax tags
- `@square` - marks operator as having same input/output structure
- `@positive_semidefinite`, `@negative_semidefinite`

**Input/output structures** use `jax.ShapeDtypeStruct` PyTrees. Operators define `in_structure()` and `out_structure()` methods.

### Domain Types

- **Stokes vectors**: `StokesI`, `StokesQU`, `StokesIQU`, `StokesIQUV` - registered JAX PyTrees for polarization data
- **Landscapes**: `HealpixLandscape`, `WCSLandscape` - sky map containers that know their geometry
- **Observations**: Lazy observation classes that load TOD data on demand

### Key Dependencies

- **JAX**: Array computation and autodiff
- **Equinox**: Immutable modules (operators are `equinox.Module`)
- **Lineax**: Linear solver framework (CG, etc.)
- **jaxtyping**: Array shape annotations
- **jax-healpy**: HEALPix sphere pixelization

## Running commands

```bash
# Use an existing virtual environment
. .venv/bin/activate

# Create an environment if needed
uv venv -p 3.12

# Install dependencies for development
uv pip install -e .[dev]

# Run tests
pytest                    # All tests
pytest -m "not slow"      # Skip slow tests
pytest tests/path/test_file.py::test_name  # Single test

# Run pre-commit hooks
prek run -a

# Linting and formatting (run after editing code)
ruff check <files>        # Check for linting errors
ruff format <files>       # Format code
```

## Guidelines

- Fix linter violations and format code after editing.
- All changes must be tested. If you're not testing your changes, you're not done.
- Get your tests to pass. If you didn't run the tests, your code does not work.
- Follow existing code style. Check neighboring files for patterns.

## Code Conventions

### Type Annotations

Use jaxtyping for array shapes:

```python
from jaxtyping import Float, Array
def func(x: Float[Array, '... n']) -> Float[Array, '... m']:
```

**Important**: Single-letter shape variables like `Float[Array, 'k']` trigger ruff F821 errors. Add a leading space to bypass: `Float[Array, ' k']`.

Use modern type annotations. Use `list`, `dict`, `tuple` directly (not from typing).

### Style

- Single quotes for strings
- Line length: 100 characters
- Operators are named with `Operator` suffix
- Config classes end with `Config`
- Private modules/functions prefixed with `_`
