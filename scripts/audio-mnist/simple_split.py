from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import argparse

# Path to the directory containing all speaker folders
# root_dir = "/network/scratch/m/mina.beiramy/bgpt/audio-mnist/data"   #change if needed
# sample_path = "/network/scratch/m/mina.beiramy/bgpt/audio-mnist/"  #change if needed

def split(root_dir, sample_path):
    base_dir = os.makedirs(sample_path, exist_ok=True)
    train_dir = os.path.join(sample_path, "train")
    test_dir = os.path.join(sample_path, "test")
    val_dir = os.path.join(sample_path, "val")

    # Make sure the directories for the splits exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Split ratios
    test_size = 0.20  # 20% for test, which makes 80% for training
    val_size = 0.25   # 25% of 20% test data will be 5% of the total data for validation

    # Function to create symlinks for the respective directories
    def create_symlinks(files, speaker_id, target_dir):
        for file in files:
            # Original filename without the speaker_id
            original_file_name = "_".join(file.split("_")[1:])
            # New filename includes the speaker_id after the label
            new_file_name = f"{file.split('_')[0]}_{speaker_id}_{original_file_name}"
            # Define the link path
            link_path = os.path.join(target_dir, new_file_name)
            # Define the target path (absolute path to the actual file)
            target_path = os.path.abspath(os.path.join(root_dir, speaker_id, file))
            # Create a symlink if it doesn't exist
            if not os.path.exists(link_path):
                os.symlink(target_path, link_path)

    # Process the files and split them
    for i in tqdm(range(1, 61), desc="Processing speakers"):
        speaker_id = f"{i:02}"
        speaker_label_dir = os.path.join(root_dir, speaker_id)
        if os.path.exists(speaker_label_dir) and os.path.isdir(speaker_label_dir):
            files = [f for f in os.listdir(speaker_label_dir) if f.endswith('.wav')]
            labels = [f.split('_')[0] for f in files]  # Extract labels for stratification

            # Perform a stratified split to maintain label proportions
            train_files, test_files, train_labels, test_labels = train_test_split(
                files, labels, test_size=test_size, stratify=labels, random_state=42
            )
            # Further split test into test and validation sets
            test_files, val_files, test_labels, val_labels = train_test_split(
                test_files, test_labels, test_size=val_size, stratify=test_labels, random_state=42
            )

            # Create symlinks for each set
            create_symlinks(train_files, speaker_id, train_dir)
            create_symlinks(test_files, speaker_id, test_dir)
            create_symlinks(val_files, speaker_id, val_dir)


def main():
    parser = argparse.ArgumentParser(description="Compress audios.")
    parser.add_argument("--root_dir", type=str, help="Specify where audio mnist data is")
    parser.add_argument("--output_path", type=str, help="Specify where audio mnist data is")

    args = parser.parse_args()
    split(args.root_dir, args.output_path)


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compress audios.")
    parser.add_argument("--root_dir", type=str, help="Specify where audio mnist data is")
    parser.add_argument("--output_path", type=str, help="Specify where audio mnist data is")

    args = parser.parse_args()

    main()
    
# python simple_split.py --root_dir /network/scratch/m/mina.beiramy/bgpt/audio-mnist/data --output_path /network/scratch/m/mina.beiramy/bgpt/audio-mnist/"