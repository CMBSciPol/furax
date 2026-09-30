# Furax

[![PyPI version](https://img.shields.io/pypi/v/furax)](https://pypi.org/project/furax/)
[![Python version](https://img.shields.io/pypi/pyversions/furax)](https://pypi.org/project/furax/)
[![License](https://img.shields.io/pypi/l/furax)](https://github.com/CMBSciPol/furax/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2603.19600-b31b1b.svg)](https://arxiv.org/abs/2603.19600)
[![Documentation Status](https://readthedocs.org/projects/furax/badge/?version=stable)](https://furax.readthedocs.io/en/stable/)
[![CI](https://github.com/CMBSciPol/furax/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/CMBSciPol/furax/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[**Docs**](https://furax.readthedocs.io/en/stable)

Furax: a Framework for Unified and Robust data Analysis with JAX.

This framework provides building blocks for solving inverse problems, in particular in the astrophysical and cosmological domains.

## Requirements

- Python 3.12 or higher
- [JAX](https://jax.readthedocs.io/en/latest/installation.html) — install separately for your target hardware (CPU, CUDA, Metal, …)

## Installation

Furax is available as [`furax`](https://pypi.org/project/furax/) on PyPI, and can be installed with:

```bash
uv add furax       # uv, recommended
pip install furax  # pip, alternative
```

> [!NOTE]
> Transposing spherical harmonic transforms (`furax.math.sht`) requires an `s2fft` fix ([#380](https://github.com/astro-informatics/s2fft/pull/380)) that is not yet released.
> Until the next release, install `s2fft` from GitHub alongside Furax:
>
> ```bash
> uv add "s2fft @ git+https://github.com/astro-informatics/s2fft"
> pip install "s2fft @ git+https://github.com/astro-informatics/s2fft"
> ```

## Developing Furax

We strongly recommend using [`uv`](https://docs.astral.sh/uv) to work on Furax.

```bash
uv sync                  # automatic editable install with `dev` dependency group
uv sync --extra <extra>  # same with additional, optional dependencies
```

We use [pytest](https://docs.pytest.org/en/stable/) for testing.
You can run the tests with:

```bash
uv run pytest
```

To ensure that your code passes the quality checks, you can use our [pre-commit](https://pre-commit.com/) configuration.
We recommend using [`prek`](https://prek.j178.dev/) to run the pre-commit hooks.

```bash
uvx prek install  # install pre-commit hooks; every commit will trigger them
uvx prek run      # run hooks on demand (staged files)
uvx prek run -a   # run hooks (all files)
```

## Citing Furax

If you use Furax in your work, please cite the [paper](https://arxiv.org/abs/2603.19600):

> Chanial et al., *Furax: A Modular JAX Framework for Linear Operators in Astrophysical and
> Cosmological Data Analysis*, arXiv:2603.19600 (2026).

Citation metadata is also available in [`CITATION.cff`](CITATION.cff).
