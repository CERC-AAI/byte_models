#!/bin/bash
#SBATCH -J train-batch
#SBATCH -o /nfs/scratch/jonathan/byte_models/logs/%x-%j.out
#SBATCH --error=/nfs/scratch/jonathan/byte_models/logs/%x-%j.err
#SBATCH -N 1
#SBATCH --gres=gpu:A100:1
#SBATCH -t 48:00:00
#SBATCH -p student

# Define the source and target directories
SRC_DIR="/home/jonathan/cerc/byte_models/bgpt"

# Change to config YAML path
CONFIG_PATH="/home/jonathan/cerc/byte_models/bgpt/configs/config_110M_image_mnist_cls_nvidia.yaml"

NUM_NODES=1
NUM_GPUS_PER_NODE=1

cd "$SRC_DIR"

source ${HOME}/.bashrc
source activate /nfs/scratch/jonathan/micromamba/envs/bgpt

# export MASTER_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_IP=localhost
# export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2001))

srun torchrun --nnodes=${NUM_NODES} --nproc_per_node=${NUM_GPUS_PER_NODE} --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29401 train-cls-distributed.py --train-config-path ${CONFIG_PATH}
# python train-gen-distributed.py --train-config-path ${CONFIG_PATH}
