import os 
import argparse
import subprocess as sb
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def _single_file(filename, abc_path, output_path):
    # abc2midi is the package that converts formats (ref: https://github.com/xlvector/abcmidi)
    command = ["abcmidi/abc2midi", os.path.join(abc_path, filename), "-o", os.path.join(output_path, filename[:-4]+".mid")]
    process = sb.run(command, stderr=sb.PIPE, stdout=sb.PIPE)

    # storing err and outs to verify file conversion warning logs if needed
    stdout_output = process.stdout.decode('utf-8')  
    stderr_output = process.stderr.decode('utf-8') 
    return stdout_output, stderr_output

# install abcmidi package first 
def convert_abc_to_midi(abc_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # check log file after conversion finishes for any errors if midi files are not working
    with open("conversion_log", "w") as log_file, Pool() as pool:
        filenames = os.listdir(abc_path)
        process_func = partial(_single_file, abc_path=abc_path, output_path=output_path)
        results = list(tqdm(pool.imap(process_func, filenames, chunksize=20), total=len(filenames), desc="Converting abc files to midi files")) # reduce chunksize for lower memory usage
        
        for stdout_output, stderr_output in results:
            log_file.write("STDOUT output:\n")
            log_file.write(stdout_output)
            log_file.write("\n\n")
            log_file.write("STDERR output:\n")
            log_file.write(stderr_output)
            log_file.write("\n\n")


def main():
    parser = argparse.ArgumentParser(description="Convert abc files to midi files.")
    parser.add_argument("abc_path", type=str, help="Path to directory containing abc files.")
    parser.add_argument("-o", "--output_path", type=str, help="Output directory for saving midi files.", required=True)
    args = parser.parse_args()

    convert_abc_to_midi(args.abc_path, args.output_path)

if __name__ == "__main__":
    main()

    # e.g. python abc2midi.py ../../data/abc_data/train -o ../../data/midi_data/train
    # e.g. python abc2midi.py ../../data/abc_data/val -o ../../data/midi_data/val
    #      python abc2midi.py abc_data/val -o midi_data/val
    