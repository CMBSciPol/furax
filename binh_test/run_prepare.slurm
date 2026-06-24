#!/bin/bash -l

#SBATCH -A mp107b
#SBATCH -C cpu
#SBATCH --qos=regular

#SBATCH --job-name=so_prepare

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --array=0-49
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yenbinhpy308@gmail.com


source .venv/bin/activate


mkdir -p slurm_logs logs


srun python binh_test/run_prepare.py