#!/bin/bash

#SBATCH --job-name=lr-finder
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=10GB
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=gpu

# apptainer shell --shell /bin/bash --nv r-4.2.1-torchcuda.sif
apptainer exec --nv /home/mjk56/palmer_scratch/yale-hpc-container/r-4.2.1-torchcuda.sif Rscript lr-finder.r
