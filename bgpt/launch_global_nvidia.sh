#!/bin/bash

SRC_DIR="/nfs/scratch/jonathan/byte_models"

# Number of jobs to submit
NUM_JOBS=3
NUM_NODES=1
NUM_GPUS_PER_NODE=4
CPUS_PER_TASK=6
MEM="32G"
TIME_LIMIT_PER_JOB="24:00:00"
# Job dependency (set to "none" for the first job)
DEPENDENCY="none"

for (( i=1; i<=NUM_JOBS; i++ ))
do
  # Job name
  JOB_NAME="continual-job-$i"

  # Submit job and capture the job ID
  if [ "$DEPENDENCY" == "none" ]; then
    JOB_SUBMIT_OUTPUT=$(sbatch --job-name=$JOB_NAME --output=${SRC_DIR}/logs/%x-%j.out --error=${SRC_DIR}/logs/%x-%j.err --time=${TIME_LIMIT_PER_JOB} --nodes=${NUM_NODES} --gres=gpu:A100:${NUM_GPUS_PER_NODE} --cpus-per-task=${CPUS_PER_TASK} --mem=${MEM} conditional_launch_nvidia.sh)
  else
    JOB_SUBMIT_OUTPUT=$(sbatch --dependency=afterany:$DEPENDENCY --job-name=$JOB_NAME --output=${SRC_DIR}/logs/%x-%j.out --error=${SRC_DIR}/logs/%x-%j.err --time=${TIME_LIMIT_PER_JOB} --nodes=${NUM_NODES} --gres=gpu:A100:${NUM_GPUS_PER_NODE} --cpus-per-task=${CPUS_PER_TASK} --mem=${MEM} conditional_launch_nvidia.sh --load-from-checkpoint)
  fi

  # Extract job ID from the submission output
  JOB_ID=$(echo $JOB_SUBMIT_OUTPUT | grep -oP '\d+')
  echo "Submitted job $JOB_ID"

  # Set this job ID as the dependency for the next job
  DEPENDENCY=$JOB_ID
done
