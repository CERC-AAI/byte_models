#!/bin/bash

# Define the source and target directories
SRC_DIR="/home/jonathan/cerc/byte_models/bgpt"

# Change to config YAML path
CONFIG_PATH="/home/jonathan/cerc/byte_models/bgpt/configs/config_110M_jon_nvidia.yaml"

cd "$SRC_DIR"

source ${HOME}/.bashrc
source activate /nfs/scratch/jonathan/micromamba/envs/bgpt

# module load rocm/5.2

export MASTER_IP=localhost
# export MASTER_IP=`ip -f inet addr show hsn0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p' | head -1`
# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29400 train-gen.py --train-config-path ${CONFIG_PATH}
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1337 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29400 train-gen-distributed.py --train-config-path ${CONFIG_PATH}
