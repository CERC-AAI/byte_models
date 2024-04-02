#!/bin/bash
#SBATCH -d afterok:1820522
#SBATCH -A CSC590
#SBATCH -J 110m-wikipedia-reloaded-1
#SBATCH -o /lustre/orion/csc590/scratch/george-adams/bgpt/logs/%x-%j.out
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 32

# Define the source and target directories
SRC_DIR="/lustre/orion/csc590/scratch/george-adams/bgpt2/byte_models/bgpt"

# Change to config YAML path
CONFIG_PATH="/lustre/orion/csc590/scratch/george-adams/bgpt2/byte_models/bgpt/configs/reload_config_110M.yaml"

cd "$SRC_DIR"

source activate /lustre/orion/csc590/scratch/george-adams/conda_envs/bgpt

module load rocm/5.2

export MASTER_IP=`ip -f inet addr show hsn0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p' | head -1`

# mkdir "$SLURM_JOB_NAME"
# mkdir "$SLURM_JOB_NAME"/checkpoints
# mkdir "$SLURM_JOB_NAME"/dataloaders

srun torchrun --nnodes=32 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29400 train-gen.py --train-config-path ${CONFIG_PATH}
