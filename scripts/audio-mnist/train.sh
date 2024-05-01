#!/bin/bash
#SBATCH -J audio-mnist-cls-btch16-patch16-epoch50-v1
#SBATCH -o ./logs/audio-mnist-cls-btch16-patch16-epoch50-v1.out
#SBATCH --error=./logs/audio-mnist-cls-btch16-patch16-epoch50-v1.err
#SBATCH -c 4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH --partition=unkillable


export WANDB_API_KEY=$(awk '/api_key/ {print $3}' /home/mila/m/mina.beiramy/.wandb_config)
module load anaconda/3
module load cudatoolkit/11.6
conda activate bgpt
python ../../bgpt/train-cls.py