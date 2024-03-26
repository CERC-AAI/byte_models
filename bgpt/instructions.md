# Important Files and Setup Guide

This document outlines the essential components and setup instructions for running batch training jobs. Key files include `config.py`, `train-gen.py`, and `launch.sh`.

## Batch Job Launching

To launch a batch job, execute:

```bash
sbatch launch.sh
```

This command starts a batch job as configured. Within `launch.sh`, the following items need modification:

1. **`SRC_DIR`**: Specify the path where the bgpt repository is located.
2. **Path to Your Conda Environment**: Update this to point to the correct conda environment.
3. **Slurm Job Name**: It's crucial to change the slurm job name because it will be used to create a directory for saving model and DataLoader checkpoints.

## Configuration (`config.py`)

Modify `config.py` to set up paths and hyperparameters:

- **`TRAIN_FOLDERS`**: Paths to folders where text files for training are located.
- **`base_path`**: The base path on which other paths will be specified.
- **`WEIGHTS_PATH`**: Central location for saving weights; maintains only one file.
- **`LOGS_PATH`**: Where loss records per epoch are stored.
- **`CHECKPOINT_PATH`**: Directory for saving model checkpoints.
- **`DATALOADER_PATH`**: Directory for saving DataLoader checkpoints.
- **Hyperparameters**: Includes `PATCH_SIZE`, `PATCH_LENGTH`, `BYTE_NUM_LAYERS`, `HIDDEN_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`, `ACCUMULATION_STEPS`, and `PATCH_SAMPLING_BATCH_SIZE`.
- **`LOAD_FROM_CHECKPOINT`**: Set to `true` to restart training from a checkpoint; `false` to start from scratch.
- **`CHECKPOINT_FREQUENCY`**: Determines how often DataLoader and model checkpoints are saved (in iterations).

## Training Generation Script (`train-gen.py`)

- **`first_launch = True`**: A hardcoded value indicating whether training is from scratch. Replace this with a flag.
- In `train_epoch()`, ensure DataLoader and model checkpoints are saved. THIS NEEDS TO BE CHECKED.
- Additionally, restarting training with the checkpointed DataLoader needs verification.

## Launching Training Command

To launch training, the command used (in `launch.sh`) is:

```bash
srun torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29400 train-gen.py
```
