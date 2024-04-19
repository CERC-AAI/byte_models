#!/bin/bash

# Define the source and target directories
SRC_DIR="/lustre/orion/csc590/scratch/george-adams/bgpt2/byte_models/bgpt"

# Change to config YAML path
CONFIG_PATH="/lustre/orion/csc590/scratch/george-adams/bgpt2/byte_models/bgpt/configs/config_110M.yaml"
NUM_NODES=32
NUM_GPUS_PER_NODE=8

cd "$SRC_DIR"

source activate /lustre/orion/csc590/scratch/george-adams/conda_envs/bgpt

module load rocm/5.2

export MASTER_IP=`ip -f inet addr show hsn0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p' | head -1`

# mkdir "$SLURM_JOB_NAME"
# mkdir "$SLURM_JOB_NAME"/checkpoints
# mkdir "$SLURM_JOB_NAME"/dataloaders

if [ "$1" = "--load-from-checkpoint" ]; then
    srun torchrun --nnodes=${NUM_NODES} --nproc_per_node=${NUM_GPUS_PER_NODE} --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29400 train-gen.py --train-config-path ${CONFIG_PATH} --load-from-checkpoint
else
    srun torchrun --nnodes=${NUM_NODES} --nproc_per_node=${NUM_GPUS_PER_NODE} --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29400 train-gen.py --train-config-path ${CONFIG_PATH}
fi