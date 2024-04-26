#!/bin/bash
#SBATCH -A CSC590
#SBATCH -J test-interactive
#SBATCH -o /lustre/orion/csc590/scratch/jonathanlimsc/bgpt/logs/%x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2

# Define the source and target directories
SRC_DIR="/ccs/home/jonathanlimsc/generalist/byte_models/bgpt"

# Change to config YAML path
CONFIG_PATH="/ccs/home/jonathanlimsc/generalist/byte_models/bgpt/configs/config_1B_jon.yaml"

cd "$SRC_DIR"

source activate /ccs/home/jonathanlimsc/.conda/envs/bgpt2

module load rocm/5.2

export MASTER_IP=`ip -f inet addr show hsn0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p' | head -1`

# mkdir "$SLURM_JOB_NAME"
# mkdir "$SLURM_JOB_NAME"/checkpoints
# mkdir "$SLURM_JOB_NAME"/dataloaders

srun torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29400 train-gen-distributed.py --train-config-path ${CONFIG_PATH}
