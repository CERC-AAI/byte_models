#!/bin/bash
# Define the source and target directories
SRC_DIR="/home/jonathan/cerc/byte_models/bgpt"

# Change to config YAML path
CONFIG_PATH="/home/jonathan/cerc/byte_models/bgpt/configs/config_110M_math_nvidia.yaml"
# TODO: Pass this through from launch_global_jon.sh
NUM_NODES=1
NUM_GPUS_PER_NODE=4

cd "$SRC_DIR"

source ${HOME}/.bashrc
source activate /nfs/scratch/jonathan/micromamba/envs/bgpt

# export MASTER_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_IP=localhost

if [ "$1" = "--load-from-checkpoint" ]; then
    srun torchrun --nnodes=${NUM_NODES} --nproc_per_node=${NUM_GPUS_PER_NODE} --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29400 train-gen.py --train-config-path ${CONFIG_PATH} --load-from-checkpoint
else
    srun torchrun --nnodes=${NUM_NODES} --nproc_per_node=${NUM_GPUS_PER_NODE} --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29400 train-gen.py --train-config-path ${CONFIG_PATH}
fi


