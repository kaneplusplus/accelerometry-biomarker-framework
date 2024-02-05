#!/bin/bash

#SBATCH --job-name=est-5-min
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=3GB
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=gpu

# apptainer shell --shell /bin/bash --nv r-4.2.1-torchcuda.sif
cd /home/mjk56/five-min-activity-match
apptainer exec --nv /home/mjk56/palmer_scratch/day-regression-5-min/r-4.2.1-torchcuda.sif Rscript --verbose estimate.r
