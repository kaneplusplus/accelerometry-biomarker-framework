#!/bin/bash

#SBATCH --job-name=proc-day
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=6GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=day

# apptainer shell --shell /bin/bash --nv r-4.2.1-torchcuda.sif
apptainer exec --nv /home/mjk56/palmer_scratch/day-regression-5-min/r-4.2.1-torchcuda.sif Rscript process-day.r
