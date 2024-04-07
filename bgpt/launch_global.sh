#!/bin/bash

# Initial setup
#SBATCH -A CSC590
#SBATCH -o /lustre/orion/csc590/scratch/george-adams/bgpt2/byte_models/bgpt/logs/%x-%j.out
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 32

SRC_DIR="/lustre/orion/csc590/scratch/george-adams/bgpt2/byte_models/bgpt"
CONFIG_PATH="/lustre/orion/csc590/scratch/george-adams/bgpt2/byte_models/bgpt/configs/config_110M.yaml"
CONDA_ENV_PATH="/lustre/orion/csc590/scratch/george-adams/conda_envs/bgpt"
MODULE_LOAD_VERSION="rocm/5.2"

# Number of jobs to submit
NUM_JOBS=n

# Job dependency (set to "none" for the first job)
DEPENDENCY="none"

for (( i=1; i<=NUM_JOBS; i++ ))
do
  # Job name
  JOB_NAME="110m-wikipedia-reloaded-$i"

  # Submit job and capture the job ID
  if [ "$DEPENDENCY" == "none" ]; then
    JOB_SUBMIT_OUTPUT=$(sbatch --job-name=$JOB_NAME --output=${SRC_DIR}/logs/%x-%j.out --time=2:00:00 --partition=batch --nodes=32 launch.sh)
  else
    JOB_SUBMIT_OUTPUT=$(sbatch --dependency=afterany:$DEPENDENCY --job-name=$JOB_NAME --output=${SRC_DIR}/logs/%x-%j.out --time=2:00:00 --partition=batch --nodes=32 conditional_launch.sh)
  fi

  # Extract job ID from the submission output
  JOB_ID=$(echo $JOB_SUBMIT_OUTPUT | grep -oP '\d+')
  echo "Submitted job $JOB_ID"

  # Set this job ID as the dependency for the next job
  DEPENDENCY=$JOB_ID
done
