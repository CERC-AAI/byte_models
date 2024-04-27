import pandas as pd
from pydub import AudioSegment
import os
import tempfile
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

def process_audio(row, chunk_dir):
    gender = row['gender']
    audio_path = row['path']
    try:
        # Load audio file
        #audio_path = os.path.join('/home/mila/m/mina.beiramy/scratch/bgpt/cv-corpus-17.0-delta-2024-03-15/en/clips', audio_path)
        audio_path = os.path.join('/home/mila/m/mina.beiramy/scratch/en/clips', audio_path)

        audio = AudioSegment.from_file(audio_path, format='mp3')
        
        # Re-encode audio to uniform bitrate before chunking
        audio = audio.set_frame_rate(8000)  # Set frame rate to 8000 Hz
        audio = audio.set_channels(1)  # Ensure mono channel
        temp_path = tempfile.mktemp('.mp3')  # Create temporary file path
        audio.export(temp_path, format="mp3", bitrate="8k")  # Export at 8k bitrate
        audio = AudioSegment.from_file(temp_path, format='mp3')  # Reload the processed audio

        # Constants for chunking and target size
        target_size_kb = 8  # Target size of each chunk in kilobytes
        target_size_bytes = target_size_kb * 1024  # Convert KB to bytes

        # Initial chunk duration estimation based on byte rate
        bytes_per_ms = len(audio.raw_data) / len(audio)
        target_duration_ms = target_size_bytes / bytes_per_ms

        # Create chunks and dynamically adjust if necessary
        chunk_data = []
        i = 0
        while i < len(audio):
            chunk = audio[i:i + int(target_duration_ms)]
            with tempfile.NamedTemporaryFile(dir=chunk_dir, delete=False, suffix='.mp3') as temp_file:
                temp_path = temp_file.name
                chunk.export(temp_path, format="mp3")

                # Check the size of the exported chunk
                file_size = os.path.getsize(temp_path)

            # Adjust the target duration based on the actual file size
            if file_size > target_size_bytes:
                target_duration_ms *= target_size_bytes / file_size  # Reduce duration
            elif file_size < target_size_bytes * 0.95:  # Allow a 5% margin below the target
                target_duration_ms *= 1.05  # Slightly increase duration

            # Rename and store chunk details
            #chunk_filename = f"{os.path.splitext(os.path.basename(audio_path))[0]}_chunk{i // int(target_duration_ms)}.mp3"
            chunk_filename = f"{os.path.splitext(os.path.basename(audio_path))[0]}_chunk_{i}.mp3"
            #print(chunk_filename)
            chunk_path = os.path.join(chunk_dir, chunk_filename)
            os.rename(temp_path, chunk_path)
            
            chunk_data.append({
                'chunk_path': chunk_filename[:-4],  # Store path without '.mp3'
                'label': gender
            })

            i += int(target_duration_ms)  # Move to the next chunk position
        
        return chunk_data
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return []

def main():
    # Load the TSV file
    #tsv_file = '/home/mila/m/mina.beiramy/scratch/bgpt/cv-corpus-17.0-delta-2024-03-15/en/validated.tsv'
    tsv_file = '/home/mila/m/mina.beiramy/scratch/en/validated.tsv'
    data = pd.read_csv(tsv_file, sep='\t')

    # Filter out rows where gender is not specified as 'male' or 'female'
    #filtered_data = data[data['gender'].isin(['male_masculine', 'female_feminine'])]
    filtered_data = data[data['gender'].isin(['male', 'female'])]

    
    # Directory for the chunks
    #chunk_dir = '/home/mila/m/mina.beiramy/scratch/bgpt/small_mozila_chunk_dir'
    chunk_dir = '/home/mila/m/mina.beiramy/scratch/bgpt/large_mozila_chunk_dir'
    if not os.path.exists(chunk_dir):
        os.makedirs(chunk_dir)
    
    # Setup multiprocessing
    with Pool(processes=4) as pool:
        partial_process_audio = partial(process_audio, chunk_dir=chunk_dir)
        chunk_lists = list(tqdm(pool.imap(partial_process_audio, [row for index, row in filtered_data.iterrows()]), total=len(filtered_data)))

    # for index, row in tqdm(filtered_data.iterrows(), total=len(filtered_data)):
    #     chunk_lists = process_audio(row=row, chunk_dir=chunk_dir)

    # Flatten the list of lists returned from processes
    flat_list = [item for sublist in chunk_lists for item in sublist]
    
    chunks_df = pd.DataFrame(flat_list)
    
    # Write the DataFrame to a TSV file
    tsv_output_path = os.path.join(chunk_dir, 'chunks_labels.tsv')
    chunks_df.to_csv(tsv_output_path, sep='\t', index=False)

    print(f"Processed {len(flat_list)} audio chunks from {len(chunk_lists)} files. Data saved to '{chunk_dir}'.")

if __name__ == "__main__":
    main()
