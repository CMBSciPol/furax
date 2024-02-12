# AstroSim

Building blocks for astrophysical inverse problems.

# Installation

```bash
git clone git@github.com:CMBSciPol/astrosim.git
cd astrosim
python3 -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

Then [Install JAX](https://jax.readthedocs.io/en/latest/installation.html) according to the target architecture.

# Testing
To check that the package is correctly installed:
```bash
pytest -s
```
