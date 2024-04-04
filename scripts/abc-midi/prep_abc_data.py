import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

def get_abc(dataset, cache_path, split, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data = load_dataset(dataset, cache_dir=cache_path, split=split)
    # Loop through the dataset and save each 'abc notation' value into separate .abc files
    for idx, sample in tqdm(enumerate(data), total=len(data), desc="Unpacking dataset into abc files"):
        abc_notation = sample['abc notation']
        file_name = f"{idx}.abc"  
        file_path = os.path.join(output_path, file_name)
        with open(file_path, 'w') as file:
            file.write(abc_notation)  



def main():
    parser = argparse.ArgumentParser(description="Download and setup abc file train-val directories.")
    parser.add_argument("dataset_id", type=str, help="Specify huggingface dataset id flag.")
    parser.add_argument("cache_dir", type=str, help="Path to download huggingface dataset.")
    parser.add_argument("--split", type=str, help="Train/Validation/Test split depending on the dataset.", required=True)
    parser.add_argument("-o", "--output_dir", type=str, help="Set output directory for saving abc files.", required=True)

    args = parser.parse_args()

    get_abc(args.dataset_id, args.cache_dir, args.split, args.output_dir)


if __name__=="__main__":
    main()

    # e.g. 
    # python prep_abc_data.py sander-wood/irishman ../../data/hf_cache --split train -o ../../data/abc_data/train
    # python prep_abc_data.py sander-wood/irishman ../../data/hf_cache --split validation -o ../../data/abc_data/val