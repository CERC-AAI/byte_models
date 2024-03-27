import argparse

from tqdm import tqdm
from pathlib import Path


def merge_write(input_files, output_dir, max_examples_per_file, max_num_output_files):
    # Open all input files
    files = [open(file, 'r') for file in input_files]

    # Store current positions in each file
    file_positions = [0] * len(files)

    reset_tracker = [True for file in input_files]

    file_idx = 0
    file_examples_idx = 0
    output_file = open(str(Path(output_dir) / f"{file_idx}.txt"), "w") 

    # Loop until all files have been reset
    while any(reset_tracker):
        if file_idx >= max_num_output_files:
            break

        for i, file in enumerate(files):
            # Print progress
            if file_idx % 10000 == 0 and file_examples_idx <= 1:
                print(f"Generating file index: {file_idx}, with {max_examples_per_file} examples per file.")

            # Move to the stored position in the file
            file.seek(file_positions[i])
            line = file.readline()
            if line:
                # Reached max_examples_per_file. Close current file, open new file
                if file_examples_idx >= max_examples_per_file:
                    output_file.close()
                    file_idx += 1
                    file_examples_idx = 0
                    if file_idx < max_num_output_files:
                        output_file = open(str(Path(output_dir) / f"{file_idx}.txt"), "w") 
                    else:
                        break

                # Write examples
                output_file.write(line)
                file_examples_idx += 1
                # Update the stored position in the file
                file_positions[i] = file.tell()
            else:
                # If the file has reached EOF, reset position to the start
                file.seek(0)
                file_positions[i] = 0
                # File marked as reset
                reset_tracker[i] = False

    # Close all input files
    for file in files:
        file.close()


# List of input file names
input_files = ['/lustre/orion/csc590/scratch/jonathanlimsc/bgpt/data/math-adder/2-digit/train.txt', 
               '/lustre/orion/csc590/scratch/jonathanlimsc/bgpt/data/math-adder/3-digit/train.txt',
               '/lustre/orion/csc590/scratch/jonathanlimsc/bgpt/data/math-adder/4-digit/train.txt',
               '/lustre/orion/csc590/scratch/jonathanlimsc/bgpt/data/math-adder/5-digit/train.txt',
               ]

# Output file name
output_dir = '/lustre/orion/csc590/scratch/jonathanlimsc/bgpt/data/math-adder/merged/train'
Path(output_dir).mkdir(parents=True, exist_ok=True)
# Will generate output files, each output file having examples interleaved from the various input files
merge_write(input_files, output_dir, max_examples_per_file=600, max_num_output_files=1000000)

print("Files generated successfully!")