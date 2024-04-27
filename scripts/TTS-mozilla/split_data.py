import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import os

def create_directories(base_path):
    for folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)

def copy_files(row, base_path, target_dir):
    # if row.iloc[0] =='chunk_path':
    #     print('**************')
    #     print(row)


    #mp3_file = f"{row['chunk_path']}.mp3"
    txt_file = f"{row['chunk_path']}.txt"
    #src_mp3 = os.path.join(base_path, mp3_file)
    src_txt = os.path.join(base_path, txt_file)
    #target_mp3 = os.path.join(base_path, target_dir, mp3_file)
    target_txt = os.path.join(base_path, target_dir, txt_file)
    #shutil.copy(src_mp3, target_mp3)
    shutil.copy(src_txt, target_txt)

def split_and_copy_files(base_path, tsv_file):
    df = pd.read_csv(tsv_file, delimiter='\t')
    print(df.head())
    print(df.iloc[0])
    # Checking label distribution and handling edge cases
    label_counts = df['label'].value_counts()
    single_instance_labels = label_counts[label_counts == 1].index.tolist()
    
    if single_instance_labels:
        # If there are labels with only one instance, drop them from stratification
        df_filtered = df[~df['label'].isin(single_instance_labels)]
    else:
        df_filtered = df

    # Perform the split
    if not df_filtered.empty:
        train_val, test = train_test_split(df_filtered, test_size=0.2, stratify=df_filtered['label'])
        train, val = train_test_split(train_val, test_size=0.25, stratify=train_val['label'])
    else:
        train, val, test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


    create_directories(base_path)
    
    # Copy files to their respective folders
    for _, row in train.iterrows():
        copy_files(row, base_path, 'train')
    for _, row in val.iterrows():
        copy_files(row, base_path, 'val')
    for _, row in test.iterrows():
        copy_files(row, base_path, 'test')

base_path = '/home/mila/m/mina.beiramy/scratch/bgpt/cv-corpus-17.0-delta-2024-03-15/chunk_dir'
tsv_file = f'{base_path}/chunks_labels.tsv'
split_and_copy_files(base_path, tsv_file)
