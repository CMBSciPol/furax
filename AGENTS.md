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
- Use jaxtyping annotations, e.g. `Inexact[jax.Array, 'dim1 dim2']`
  - for uni-dimensional arrays, prepend a space to the start of the shape (e.g. `Float32[jax.Array, ' x']`) to turn Ruff F821 error (undefined name) into F722 (syntax error in forward annotation, ignored)

## Testing

- Add tests for new behaviour: cover success, failure, and edge cases.
- Use `@pytest.mark.parametrize` for multiple similar inputs: consolidate tests that only differ in input/expected values into a single parametrized test.
- Use `@pytest.mark.slow` for expensive tests (excluded from default test runs)
- Top-level `tests/conftest.py` contains an session-scope autouse fixture that sets `jax_enable_x64=True` for all tests.

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
- Dataclass fields can be dynamic (JAX arrays, traced) or static (shapes, metadata, etc.). Mask static fields with `axis: int = field(metadata={'static': True})`
- Subclasses must implement at least `mv(x)` (matrix-vector product)
- Operators are directly callable: `op(x) = op.mv(x)`
- Properties: `.T` (transpose), `.I` (inverse), `in_structure`/`out_structure` (`PyTree[jax.ShapeDtypeStruct]`, static)
- Tag system: `OperatorTag` IntFlag marks algebraic properties, applied via decorators: `@square`, `@symmetric`, `@orthogonal`, `@diagonal`, etc.
- Composite operators:
  - `op1 @ op2 = CompositionOperator(op1, op2)`
  - `op1 + op2 = AdditionOperator(op1, op2)`
- Composite operators can be simplified (`op.reduce()`) using algebraic rules from `COMPOSITION_RULE_REGISTRY` and `ADDITION_RULE_REGISTRY` (e.g. `A @ A.I -> I`)

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
