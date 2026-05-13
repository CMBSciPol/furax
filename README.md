# Furax

[![PyPI version](https://badge.fury.io/py/furax.svg)](https://badge.fury.io/py/furax)
[![Python version](https://img.shields.io/pypi/pyversions/furax)](https://pypi.org/project/furax/)
[![Documentation Status](https://readthedocs.org/projects/furax/badge/?version=latest)](https://furax.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/CMBSciPol/furax/actions/workflows/ci.yml/badge.svg)](https://github.com/CMBSciPol/furax/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[**Docs**](https://furax.readthedocs.io/en/stable)

Furax: a Framework for Unified and Robust data Analysis with JAX.

This framework provides building blocks for solving inverse problems, in particular in the astrophysical and cosmological domains.

## Requirements

- Python >= 3.11
- [JAX](https://jax.readthedocs.io/en/latest/installation.html) — install separately for your target hardware (CPU, CUDA, Metal, …)

## Installation

Furax is available as [`furax`](https://pypi.org/project/furax/) on PyPI, and can be installed with:

```bash
pip install furax
```

### Development version

Clone the repository, and navigate to the root directory of the project.
For example:

```bash
git clone git@github.com:CMBSciPol/furax.git
cd furax
```

Then, install the package with:

```bash
pip install .
```

## Developing Furax

After cloning, install in editable mode and with development dependencies:

```bash
pip install -e .[dev]
```

We use [pytest](https://docs.pytest.org/en/stable/) for testing.
You can run the tests with:

```bash
pytest
```

To ensure that your code passes the quality checks,
you can use our [pre-commit](https://pre-commit.com/) configuration:

1. Install the pre-commit hooks with

```bash
pre-commit install
```

2. That's it! Every commit will trigger the code quality checks.
