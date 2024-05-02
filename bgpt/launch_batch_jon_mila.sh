#!/bin/bash
#SBATCH -J train-batch
#SBATCH -o /network/scratch/m/mina.beiramy/bgpt_shared/logs/%x-%j.out
#SBATCH --error=/network/scratch/m/mina.beiramy/bgpt_shared/logs/%x-%j.err
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 24:00:00
#SBATCH --partition=unkillable

# Define the source and target directories
SRC_DIR="/home/mila/j/jonathan.lim/cerc/byte_models/bgpt"

# Change to config YAML path
CONFIG_PATH="/home/mila/j/jonathan.lim/cerc/byte_models/bgpt/configs/config_7M_multimodal_mnist.yaml"

NUM_NODES=1
NUM_GPUS_PER_NODE=1

cd "$SRC_DIR"

module load anaconda/3
module load cudatoolkit/11.6
conda activate bgpt3

# export MASTER_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_IP=localhost
export WANDB_API_KEY=$(awk '/api_key/ {print $3}' /home/mila/j/jonathan.lim/.wandb_config)
# export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2001))

# srun torchrun --nnodes=${NUM_NODES} --nproc_per_node=${NUM_GPUS_PER_NODE} --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29401 train-gen-distributed.py --train-config-path ${CONFIG_PATH}
python train-gen-distributed.py --train-config-path ${CONFIG_PATH}
