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
- `uvx prek ruff-check`: Run linting hook
- `uvx prek ruff-format`: Run formatting hook
- `uvx prek mypy`: Run type-checking hook

## Guidelines

- Before adding a new operator or test, read an existing one in the same module and mirror its patterns.
- Always run `uvx prek --files <files>` when you are done with your task, and address errors.
- Naming: `PascalCase` for classes/operators (e.g. `BeamOperator`), `snake_case` for functions/variables. Single-quote strings, 100-char lines (enforced by ruff).
- Write Google-style docstrings.
- Use comments purposefully. Do not narrate code. Explain invariants and unusual patterns.
- Imports at top of file. Valid exceptions: circular imports, lazy loading.
- Use jaxtyping annotations, e.g. `Inexact[jax.Array, 'dim1 dim2']`
  - for uni-dimensional arrays, prepend a space to the start of the shape (e.g. `Float32[jax.Array, ' x']`) to turn Ruff F821 error (undefined name) into F722 (syntax error in forward annotation, ignored)
  - Enable runtime checks with beartype: `uv run pytest --jaxtyping-packages=furax,beartype.beartype(...)` (commented config in `pyproject.toml`)
- Prefer `furax.tree` over `jax.tree` and `jax.tree_util`. In particular use the elementwise helpers (`tree.add`, `tree.sub`, `tree.mul`, `tree.dot`, `tree.zeros_like` etc.) instead `jax.tree.map(jnp.add, ...)` and friends.
- Call operators directly (`op(x)`), not `op.mv(x)`. Reserve `mv` for the method definition in an operator subclass.
- Use new-style typed key arrays `jax.random.key()` instead of legacy uint32 `jax.random.PRNGKey()`.

## When to ask first

- Don't add or remove dependencies (`uv add`/`uv remove`) without confirming.
- Don't weaken checks to go green: no blanket `# type: ignore` / `# noqa` or relaxing ruff/mypy config — fix the cause or ask.
- Don't change operator algebra (tags, composition/addition reduction rules) or public API signatures without confirming.
- If a task needs tools or permissions beyond what's available, stop and ask rather than guess.

## Testing

- Add tests for new behaviour: cover success, failure, and edge cases.
- Use `@pytest.mark.parametrize` for multiple similar inputs: consolidate tests that only differ in input/expected values into a single parametrized test.
- Use `@pytest.mark.slow` for expensive tests (excluded from default test runs)
- Top-level `tests/conftest.py` contains an session-scope autouse fixture that sets `jax_enable_x64=True` for all tests.
- x64-off tests: mark `@pytest.mark.insubprocess` and flip `jax.config.update('jax_enable_x64', False)` in-body (the autouse fixture forces x64 on otherwise). Precedent: `tests/core/base/test_inverse.py`.

## Architecture

- /src/furax.................Furax source code
- /src/furax/core............Linear operator algebra
- /src/furax/tree.py.........PyTree utilities
- /src/furax/mapmaking.......Mapmaking-specific code
- /src/furax/interfaces......Mapmaking interfaces with sotodlib/toast/litebird_sim
- /src/so_mapmaking..........CLI for multi-observation mapmaking with Simons Observatory data
- /tests.....................Tests (mirror /src structure)

### AbstractLinearOperator (`/src/furax/core/_base.py`)

`AbstractLinearOperator` (a frozen dataclass ABC) is the base class for all linear operators. Key features:

- Automatic PyTree dataclass: subclasses get `@dataclass(frozen=True)` and registered as JAX PyTree nodes via `__init_subclass__`
- Dataclass fields can be dynamic (JAX arrays, traced) or static (shapes, metadata, etc.). Mark static fields with `axis: int = field(metadata={'static': True})`
- Subclasses must implement at least `mv(x)` (matrix-vector product)
- Operators are directly callable: `op(x) = op.mv(x)`
- Properties: `.T` (transpose), `.I` (inverse), `in_structure`/`out_structure` (`PyTree[jax.ShapeDtypeStruct]`, static)
- Tag system: `OperatorTag` IntFlag marks algebraic properties, applied via decorators: `@square`, `@symmetric`, `@orthogonal`, `@diagonal`, etc.
- Composite operators:
  - `op1 @ op2 = CompositionOperator(op1, op2)`
  - `op1 + op2 = AdditionOperator(op1, op2)`
- Composite operators can be simplified (`op.reduce()`) using algebraic rules from `COMPOSITION_RULE_REGISTRY` and `ADDITION_RULE_REGISTRY` (e.g. `A @ A.I -> I`)

### Observation operators (`/src/furax/obs`)

- `PointingOperator`: HEALPix sky map ⇄ TOD via boresight + detector quaternions (on-the-fly)
- `HWPOperator`, `LinearPolarizerOperator`, `QURotationOperator`: polarization modulation
- `AbstractSEDOperator` → `DustOperator` / `SynchrotronOperator` / `CMBOperator`: SED operators for component separation
- `StokesLandscape` (`landscapes.py`): Stokes-aware (I/Q/U) HEALPix sky pixelisation
- `Stokes` and I/QU/IQU/IQUV variants (`stokes.py`): single-array Stokes containers (components stacked on the leading axis)

### Mapmaking pipeline (`/src/furax/mapmaking`)

- `acquisition.py` → `build_acquisition_operator`: builds the acquisition operator `A` from observation metadata
- `mapmaker.py`: `BinnedMapMaker` / `MLMapmaker` / `ATOPMapMaker`; solve `AᵀN⁻¹A x = AᵀN⁻¹d` via lineax CG
- `noise.py`: noise-model operators
- `config.py` / `_model.py`: apischema-driven YAML configuration
- `preconditioner.py` / `templates.py`: PCG infrastructure

### Interfaces (`/src/furax/interfaces`)

- `lineax.py` → `as_lineax_operator`: wrap a furax operator as a lineax `LinearOperator`
- `sotodlib/`, `toast/`, `litebird_sim/`: SO / TOAST / LiteBird adapters
- CLI entrypoints: `furax-so-atomic-map` (`sotodlib.mapmaker:main_cli`), `furax-so-prepare` / `furax-so-map` (`so_mapmaking`)

### Mapmaking conventions

- `double_precision=False` → float32 on every float field, including geometry (timestamps, HWP angles, quaternions); the pipeline then runs under `jax_enable_x64=False`, where float64 arrays are illegal. Timestamps are an exception in form only: the reader rebases them to a per-observation zero origin (in float64, before the downcast) so the absolute POSIX epoch does not exhaust the float32 range. The pipeline uses only time differences, so this is exact; absolute UTC is still read from the interface where needed (e.g. pointing).

## Designing operators from math

The repo-local **furax-operator-designer** skill (`.claude/skills/furax-operator-designer/`) turns a
mathematical description (LaTeX, PDF, or Markdown) into a verified furax operator, a design doc with
matrix sketches, and an executed example notebook. Reach for it when implementing an operator from
equations.

Minimal example — from the LaTeX of a forward-difference operator $D:\mathbb{R}^n\to\mathbb{R}^{n-1}$:

$$ (D x)_i = x_{i+1} - x_i $$

it writes the operator

```python
class ForwardDifferenceOperator(AbstractLinearOperator):
    def mv(self, x):
        return x[1:] - x[:-1]
```

and verifies it against the math via `as_matrix()` (shown for $n = 3$ — note the bidiagonal
$-1/+1$ structure matches the equation):

```python
>>> op = ForwardDifferenceOperator(in_structure=jax.ShapeDtypeStruct((3,), jnp.float64))
>>> op.as_matrix()
Array([[-1.,  1.,  0.],
       [ 0., -1.,  1.]], dtype=float64)
```
