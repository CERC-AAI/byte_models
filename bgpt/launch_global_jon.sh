#!/bin/bash

SRC_DIR="/lustre/orion/csc590/scratch/jonathanlimsc/bgpt"

# Number of jobs to submit
NUM_JOBS=2
NUM_NODES=2

# Job dependency (set to "none" for the first job)
DEPENDENCY="none"

for (( i=1; i<=NUM_JOBS; i++ ))
do
  # Job name
  JOB_NAME="test-continual-$i"

  # Submit job and capture the job ID
  if [ "$DEPENDENCY" == "none" ]; then
    JOB_SUBMIT_OUTPUT=$(sbatch -A csc590 --job-name=$JOB_NAME --output=${SRC_DIR}/logs/%x-%j.out --time=00:05:00 --partition=batch --nodes=${NUM_NODES} conditional_launch_jon.sh)
  else
    JOB_SUBMIT_OUTPUT=$(sbatch -A csc590 --dependency=afterany:$DEPENDENCY --job-name=$JOB_NAME --output=${SRC_DIR}/logs/%x-%j.out --time=00:05:00 --partition=batch --nodes=${NUM_NODES} conditional_launch_jon.sh --load-from-checkpoint)
  fi

  # Extract job ID from the submission output
  JOB_ID=$(echo $JOB_SUBMIT_OUTPUT | grep -oP '\d+')
  echo "Submitted job $JOB_ID"

  # Set this job ID as the dependency for the next job
  DEPENDENCY=$JOB_ID
done
