import cv2
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path

def unpickle(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data

def write_samples_to_file(x, y, output_dir, output_file_extension, offset_index=0):
    num_samples = x.shape[0]
    print(f"Writing {num_samples} samples to {output_dir} in .{output_file_extension} format")
    for i in tqdm(range(num_samples)):
        sample_x = x[i]
        sample_y = y[i]
        # Filenaming should be <label>_<name>.<file-extension> for bgpt classification
        output_filepath = f"{output_dir}/{sample_y}_{offset_index+i}.{output_file_extension}"
        cv2.imwrite(output_filepath, sample_x)
    
    return num_samples

def list_files_in_directory(directory):
    # Create a Path object for the directory
    dir_path = Path(directory)
    output_list = []
    # Use the glob() method to get a list of all files in the directory
    files_list = dir_path.glob('*')
    
    # Iterate over the files and print their names
    for file in files_list:
        if file.is_file():
            output_list.append(str(dir_path / file.name))
    
    return output_list

def main(args):
    print(args)

    data_filepaths = list_files_in_directory(args.input_dir)
    output_dir = args.output_dir
    output_file_extension = args.output_file_extension
    image_height = args.image_height
    image_width = args.image_width
    num_channels = args.image_channels
    image_dim = image_height * image_width

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    offset_index = 0

    for i, data_filepath in enumerate(data_filepaths):
        print(f"Reading {data_filepath}")
        d = unpickle(data_filepath)
        x = d['data']
        y = d['labels']
        y = [i-1 for i in y]
        num_samples = x.shape[0]
        x = np.dstack((x[:, :image_dim], x[:, image_dim:2*image_dim], x[:, 2*image_dim:]))
        x = x.reshape((num_samples, image_height, image_width, num_channels))

        x = x[0:num_samples, :, :, :]
        y = y[0:num_samples]

        index = write_samples_to_file(x, y, output_dir, output_file_extension, offset_index=offset_index)
        offset_index += index
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image files given the raw ImageNet pickled files downloaded from https://image-net.org/download-images")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the input ImageNet files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write output image files to")
    parser.add_argument("--output_file_extension", type=str, default="jpeg", help="Output file extension. Defaults to jpeg.")
    parser.add_argument("--image_height", type=int, default=64, help="Height of image")
    parser.add_argument("--image_width", type=int, default=64, help="Width of image")
    parser.add_argument("--image_channels", type=int, default=3, help="Number of image channels")
    args = parser.parse_args()

    main(args)