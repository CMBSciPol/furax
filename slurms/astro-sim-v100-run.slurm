#!/bin/bash
#SBATCH --account=nih@v100
#SBATCH --job-name=astro-sim-run # Name of job
# Other partitions are usable by activating/uncommenting
# one of the 5 following directives:
#SBATCH -C v100-16g                 # decommenter pour reserver uniquement des GPU V100 16 Go
##SBATCH -C v100-32g                 # decommenter pour reserver uniquement des GPU V100 32 Go
##SBATCH --partition=gpu_p2          # decommenter pour la partition gpu_p2 (GPU V100 32 Go)
##SBATCH -C a100                     # decommenter pour la partition gpu_p5 (GPU A100 80 Go)
# Ici, reservation de 8x10=80 CPU (4 taches par noeud) et de 8 GPU (4 GPU par noeud) sur 2 noeuds :
#SBATCH --nodes=1                  # nombre de noeud
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1      # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1               # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)

# Le nombre de CPU par tache doit etre adapte en fonction de la partition utilisee. Sachant
# qu'ici on ne reserve qu'un seul GPU par tache (soit 1/4 ou 1/8 des GPU du noeud suivant
# la partition), l'ideal est de reserver 1/4 ou 1/8 des CPU du noeud pour chaque tache:
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (un quart du noeud ici)
##SBATCH --cpus-per-task=3           # nombre de CPU par tache pour gpu_p2 (1/8 du noeud 8-GPU)
##SBATCH --cpus-per-task=6           # nombre de CPU par tache pour gpu_p4 (1/8 du noeud 8-GPU)
##SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p5 (1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=01:00:00              # maximum execution time requested (HH:MM:SS)
## Use this if your code runs fast or you are in dev phase
##SBATCH --qos=qos_gpu-dev            # quality of service (QOS) dev
#SBATCH --output=astro-sim-run.out # name of output file
#SBATCH --error=astro-sim-run.err  # name of error file 
# Cleans out modules loaded in interactive and inherited by default
module purge

# Uncomment the following module command if you are using the "gpu_p5" partition
# to have access to the modules compatible with this partition.
#module load cpuarch/amd

# Loading modules
module load python/3.10.4
source venv/bin/activate

module load cuda/11.8.0  cudnn/8.9.7.29-cuda 
# Echo of launched commands

set -x
# For the "gpu_p5" partition, the code must be compiled with the compatible modules.
# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun python $1
