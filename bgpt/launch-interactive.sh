#!/bin/bash

USERNAME=jonathanlimsc
# Define the source and target directories
SRC_DIR=/lustre/orion/csc590/scratch/${USERNAME}/bgpt

# Change to config YAML path
CONFIG_PATH=$(pwd)/configs/config_100M.yaml

cd "$SRC_DIR"

source /ccs/home/jonathanlimsc/miniconda3/etc/profile.d
conda activate /ccs/home/jonathanlimsc/.conda/envs/bgpt/

module load rocm/5.2

# export MASTER_IP=`ip -f inet addr show hsn0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p' | head -1`

# mkdir "$SLURM_JOB_NAME"
# mkdir "$SLURM_JOB_NAME"/checkpoints
# mkdir "$SLURM_JOB_NAME"/dataloaders

python train-gen.py --train-config-path ${CONFIG_PATH}
