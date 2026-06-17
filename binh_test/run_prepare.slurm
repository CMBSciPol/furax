#!/bin/bash -l

#SBATCH -A mp107b
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --array=0-13
#SBATCH --job-name=so_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yenbinhpy308@gmail.com

source .venv/bin/activate

mkdir -p slurm_logs

srun python binh_test/run_prepare.py