#!/bin/bash

# Define the source and target directories
SRC_DIR="/ccs/home/jonathanlimsc/generalist/byte_models/bgpt"

# Change to config YAML path
CONFIG_PATH="/ccs/home/jonathanlimsc/generalist/byte_models/bgpt/configs/config_110M_jon.yaml"

cd "$SRC_DIR"

source /ccs/home/jonathanlimsc/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/jonathanlimsc/.conda/envs/bgpt2

module load rocm/5.2

export MASTER_IP=`ip -f inet addr show hsn0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p' | head -1`

# mkdir "$SLURM_JOB_NAME"
# mkdir "$SLURM_JOB_NAME"/checkpoints
# mkdir "$SLURM_JOB_NAME"/dataloaders

srun torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29400 train-gen.py --train-config-path ${CONFIG_PATH}
