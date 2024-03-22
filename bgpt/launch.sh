#!/bin/bash
 #SBATCH -A CSC590
 #SBATCH -J byte-models-test
 #SBATCH -o /lustre/orion/csc590/scratch/george-adams/bgpt/logs/%x-%j.out
 #SBATCH -t 2:00:00
 #SBATCH -p batch
 #SBATCH -N 1


cd /lustre/orion/csc590/scratch/george-adams/bgpt

source activate /lustre/orion/csc590/scratch/george-adams/conda_envs/bgpt

module load rocm/5.2

torchrun --nproc_per_node=8 train-gen.py
