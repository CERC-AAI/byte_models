#!/bin/bash
#SBATCH -J inference
#SBATCH -o /nfs/scratch/jonathan/byte_models/logs/%x-%j.out
#SBATCH --error=/nfs/scratch/jonathan/byte_models/logs/%x-%j.err
#SBATCH -N 1
#SBATCH --gres=gpu:A100:1
#SBATCH -t 2:00:00
#SBATCH -p student

# Define the source and target directories
SRC_DIR="/home/jonathan/cerc/byte_models/bgpt"

# Change to config YAML path
CONFIG_PATH="/home/jonathan/cerc/byte_models/bgpt/configs/config_110M_inference_nvidia.yaml"

cd "$SRC_DIR"

source ${HOME}/.bashrc
source activate /nfs/scratch/jonathan/micromamba/envs/bgpt

python inference.py --config-path ${CONFIG_PATH}
