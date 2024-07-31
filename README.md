# Furax

Furax: a Framework for Unified and Robust data Analysis with JAX.

This framework provides building blocks for solving inverse problems, in particular in the astrophysical and cosmological domains.

## Installation of the stable packaged distribution

The GitLab repository is currently private and two pieces of information need to be specified: the URL of the repository and the credentials.

- Create a Personal Access Token by following these [instructions](https://gitlab.in2p3.fr/help/user/profile/personal_access_tokens).
For the name, choose something like `read scipol` and for the scopes, select `read_api`.

- Edit the file ~/.netrc to store the credentials
```
machine gitlab.in2p3.fr
login __token__
password <your-personal-access-token>
```
Make sure the .netrc file are appropriately set:
```commandline
chmod 600 ~/.netrc
```
- Edit ~/.config/pip/pip.conf
```
[global]
index-url = https://gitlab.in2p3.fr/api/v4/projects/26092/packages/pypi/simple
```
- After these steps are completed: you can install `furax` in the normal way:
```commandline
pip install furax
```

## Installation for development purposes

```bash
git clone git@github.com:CMBSciPol/furax.git
cd furax
python3 -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

Then [Install JAX](https://jax.readthedocs.io/en/latest/installation.html) according to the target architecture.

### Testing
To check that the package is correctly installed:
```bash
pytest
```

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
