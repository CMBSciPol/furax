# Furax repository

Python, JAX, linear operator framework, CMB mapmaking.

## Commands

- `uv run path/to/script.py|command`: Run a Python file/snippet using the project environment
- `uv add <package>`: Add a new dependency to the project
- `uv remove <package>`: Remove a dependency from the project
- `uv run pytest -v`: Run the full test suite
- `uv run pytest path/to/test.py -v`: Run a single test file
- `uv run pytest path/to/test.py::TestClass::test_method -v`: Run a single test
- `uvx prek run`: Run pre-commit hooks (staged files)
- `uvx prek run -a`: Run pre-commit hooks (all files)
- `uvx prek ruff-check --files <files>`: Lint these files
- `uvx prek ruff-format --files <files>`: Format these files
- `uvx prek mypy --files <files>`: Type-check these files

## Guidelines

- Follow existing code style.
- Use comments purposefully. Do not narrate code. Explain invariants and unusual patterns.
- Always format and check Python files after writing or editing them.
- Imports at top of file. Valid exceptions: circular imports, lazy loading.

## Testing

- Add tests for new behaviour: cover success, failure, and edge cases.
- Use `@pytest.mark.parametrize` for multiple similar inputs: consolidate tests that only differ in input/expected values into a single parametrized test.
- Top-level `tests/conftest.py` contains an session-scope autouse fixture that sets `jax_enable_x64=True` for all tests.

## Architecture

- /src/furax.................Furax source code
- /src/furax/core............Linear operator algebra
- /src/furax/mapmaking.......Mapmaking-specific code
- /src/furax/interfaces......Mapmaking interfaces with sotodlib/toast/litebird_sim
- /src/so_mapmaking..........CLI for multi-observation mapmaking with Simons Observatory data
- /tests.....................Tests (mirror /src structure)
