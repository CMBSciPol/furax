# CLAUDE.md

Furax provides composable building blocks for solving inverse problems in astrophysical and cosmological domains.

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
