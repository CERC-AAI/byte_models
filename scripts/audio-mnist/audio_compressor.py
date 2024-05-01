import os
from pydub import AudioSegment
from multiprocessing import Pool
from tqdm import tqdm
import argparse

def reduce_file_size(input_file, base_directory):
    # Define the new directory structure by appending '_r' to the original directory name
    new_dir = os.path.join(base_directory, os.path.dirname(input_file).split(os.sep)[-1] + '_r')
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Define the output file path in the new directory
    output_file = os.path.join(new_dir, os.path.basename(input_file).replace(".wav", "_compressed.mp3"))

    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Initial resampling to a lower sample rate
    target_sample_rate = 8000
    if audio.frame_rate > target_sample_rate:
        audio = audio.set_frame_rate(target_sample_rate)

    # Compress the audio by reducing the bitrate and using a compressed format
    parameters = {
        "bit_rate": "32k",  # Initial bitrate
        "codec": "libmp3lame",
        "parameters": ["-ac", "1"]  # Mono channel
    }

    # Try reducing the bitrate to decrease file size
    while True:
        # Export the audio to check the file size
        audio.export(output_file, format="mp3", bitrate=parameters["bit_rate"], codec=parameters["codec"], parameters=parameters["parameters"])
        
        # Check the output file size
        file_size_kb = os.path.getsize(output_file) / 1024
        if file_size_kb <= 8:
            break
        elif parameters["bit_rate"] == "8k":
            break
        else:
            # Reduce bitrate further if the file size is still too large
            if parameters["bit_rate"] == "32k":
                parameters["bit_rate"] = "16k"
            elif parameters["bit_rate"] == "16k":
                parameters["bit_rate"] = "8k"
    
    print('done, sampled')

def find_audio_files(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    return audio_files

def main():
    parser = argparse.ArgumentParser(description="Compress audios.")
    parser.add_argument("--base_dir", type=str, help="Specify where audio mnist data is")

    kargs = parser.parse_args()
    #base_directory = "/network/scratch/m/mina.beiramy/bgpt/audio-mnist"  # Set this to the base directory where your 'train', 'test', 'val' folders are located
    directories = [os.path.join(kargs.base_dir, d) for d in ["train", "test", "val"]]  # Directories containing audio files
    all_audio_files = []
    for directory in directories:
        all_audio_files.extend(find_audio_files(directory))

    # Process files in parallel
    args = [(file, kargs.base_dir) for file in all_audio_files]
    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.starmap(reduce_file_size, args), total=len(all_audio_files), desc="Processing audio files"))

if __name__ == "__main__":
    main()


# python audio_compressor.py --base_dir /network/scratch/m/mina.beiramy/bgpt/audio-mnist