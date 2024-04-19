#!/bin/bash

SRC_DIR="/lustre/orion/csc590/scratch/george-adams/bgpt2/byte_models/bgpt"

# Number of jobs to submit
NUM_JOBS=2
NUM_NODES=32
NUM_GPUS_PER_NODE=8
TIME_LIMIT_PER_JOB="2:00:00"

# Job dependency (set to "none" for the first job)
DEPENDENCY="none"

for (( i=1; i<=NUM_JOBS; i++ ))
do
  # Job name
  JOB_NAME="110m-wikipedia-reloaded-$i"

  # Submit job and capture the job ID
  if [ "$DEPENDENCY" == "none" ]; then
    JOB_SUBMIT_OUTPUT=$(sbatch --job-name=$JOB_NAME --output=${SRC_DIR}/logs/%x-%j.out --error=${SRC_DIR}/logs/%x-%j.err --time=${TIME_LIMIT_PER_JOB} --nodes=${NUM_NODES} --partition=batch conditional_launch.sh
  else
    JOB_SUBMIT_OUTPUT=$(sbatch --dependency=afterany:$DEPENDENCY --job-name=$JOB_NAME --output=${SRC_DIR}/logs/%x-%j.out --error=${SRC_DIR}/logs/%x-%j.err --time=${TIME_LIMIT_PER_JOB} --nodes=${NUM_NODES} --partition=batch conditional_launch.sh --load-from-checkpoint)
  fi

  # Extract job ID from the submission output
  JOB_ID=$(echo $JOB_SUBMIT_OUTPUT | grep -oP '\d+')
  echo "Submitted job $JOB_ID"

  # Set this job ID as the dependency for the next job
  DEPENDENCY=$JOB_ID
done
