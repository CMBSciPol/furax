# Furax

Furax: a Framework for Unified and Robust data Analysis with JAX.

This framework provides building blocks for solving inverse problems, in particular in the astrophysical and cosmological domains.

## Installation

Start by installing [`JAX`](https://jax.readthedocs.io/en/latest/installation.html) for the target architecture.
We recommend using a virtual environment of your choice (e.g. `venv`, `conda`).

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

## Running on JeanZay

### Load cuda and and cudnn for JAX

```bash
module load cuda/11.8.0 cudnn/8.9.7.29-cuda
```

### Create Python env (only the first time)

```bash
module load python/3.10.4 && conda deactivate
python -m venv venv
source venv/bin/activate
# install jax
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# install furax
pip install -e .[dev]
```

### launch script

To launch only the pytests

```bash
sbatch slurms/astro-sim-v100-testing.slurm
```

To launch your own script

```bash
sbatch slurms/astro-sim-v100-run.slurm yourscript.py
```

You can also allocate ressources and go into bash mode

```bash
srun --pty --account=nih@v100 --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread bash
module purge
module load python/3.10.4
source venv/bin/activate
module load cuda/11.8.0  cudnn/8.9.7.29-cuda
# Then do your thing
python my_script.py
pytest
```

Don't leave the bash running !! (I would suggest running script with sbatch)

### Specific for nih / SciPol project

The repo is already in the commun WORK folder, the data is downloaded and the environment is ready.

You only need to do this

```bash
cd $ALL_CCFRWORK/furax-main
```

Then launch scripts as you see fit
