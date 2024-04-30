from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import argparse
from tqdm import tqdm
from pathlib import Path

def main(args):
    print(args)

    output_dir = args.output_dir
    output_file_extension = args.output_file_extension
    validation_size = args.validation_size
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_dir = Path(output_dir) / 'train'
    val_dir = Path(output_dir) / 'validation'
    test_dir = Path(output_dir) / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset("mnist")

    train_size = len(dataset['train'])  
    val_index = train_size - validation_size
    test_size = len(dataset['test'])
    print(f"Writing {train_size} training examples to {train_dir}")
    print(f"Writing {validation_size} training examples to {val_dir}")

    for i, row in tqdm(enumerate(dataset['train']), total=len(dataset['train'])):
        img_obj = row['image']
        label = row['label']
        filename = f"{label}_{i}.{output_file_extension}"
        if i < val_index:
            write_path = str(train_dir / filename)
        else:
            write_path = str(val_dir / filename)
        
        img_obj.save(write_path)

    print(f"Writing {test_size} training examples to {test_dir}")
    for i, row in tqdm(enumerate(dataset['test']), total=len(dataset['test'])):
        img_obj = row['image']
        label = row['label']
        filename = f"{label}_{i}.{output_file_extension}"
        
        write_path = str(test_dir / filename)
        img_obj.save(write_path)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image files downloaded from https://huggingface.co/datasets/mnist")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write output image files to")
    parser.add_argument("--validation_size", type=int, default=1000, help="Number of validation examples to take from training dataset")
    parser.add_argument("--output_file_extension", type=str, default="bmp", help="Output file extension. Defaults to bmp.")
    args = parser.parse_args()

    main(args)