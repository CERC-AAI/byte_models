import os
import math
import time
import json
import torch
import random
import numpy as np
import yaml
import wandb
import argparse

from copy import deepcopy
from utils import *
# from config import *
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, get_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Set up distributed training
world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0

if world_size > 1:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl') if world_size > 1 else None
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
# Set random seed
seed = 0 + global_rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
def find_most_recent_file_from_checkpoint_info(checkpoint_info_filepath):
    checkpoint_filepath = None
    with open(checkpoint_info_filepath, "r") as f:
        data = json.load(f)
        checkpoint_filepath = data.get("latest_checkpoint_path", None)

    return checkpoint_filepath

def collate_batch(input_patches):

    input_patches, labels = zip(*input_patches)
    input_patches = torch.nn.utils.rnn.pad_sequence(input_patches, batch_first=True, padding_value=256)
    labels = torch.stack(labels, dim=0)

    return input_patches.to(device), labels.to(device)

def list_files_in_directory(directories):
    file_list = []
    
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list

def read_bytes(filename,
               patch_size,
               ):
    
    ext = filename.split('.')[-1]
    ext = bytearray(ext, 'utf-8')
    ext = [byte for byte in ext][:patch_size]

    with open(filename, 'rb') as f:
        file_bytes = f.read()

    bytes = []
    for byte in file_bytes:
        bytes.append(byte)

    if len(bytes)%patch_size!=0:
        bytes = bytes + [256] * (patch_size - len(bytes) % patch_size)

    bos_patch = ext + [256] * (patch_size - len(ext))
    bytes = bos_patch + bytes + [256] * patch_size

    return bytes

class ByteDataset(Dataset):
    def __init__(self, filenames, patch_size, patch_length):
        self.filenames = []
        self.labels = {}
        self.patch_size = patch_size
        self.patch_length = patch_length

        for filename in tqdm(filenames):
            file_size = os.path.getsize(filename)
            file_size = math.ceil(file_size / patch_size)
            ext = filename.split('.')[-1]
            label = os.path.basename(filename).split('_')[0]
            label = f"{label}.{ext}"

            if file_size <= patch_length-2:
                self.filenames.append((filename, label))
                if label not in self.labels:
                    self.labels[label] = len(self.labels)
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        filename, label = self.filenames[idx]
        file_bytes = read_bytes(filename, self.patch_size)

        file_bytes = torch.tensor(file_bytes, dtype=torch.long)
        label = torch.tensor(self.labels[label], dtype=torch.long)
        
        return file_bytes, label


# call model with a batch of input
def process_one_batch(model, batch, loss_fn):
    input_patches, labels = batch
    logits = model(input_patches)
    loss = loss_fn(logits, labels)
    prediction = torch.argmax(logits, dim=1)
    acc_num = torch.sum(prediction==labels)

    return loss, acc_num

# do one epoch for training
def train_epoch(model,
                train_set,
                lr_scheduler,
                scaler,
                optimizer,
                loss_fn,
                epoch,
                batch_size,
                logging_frequency,
                global_total_iters=1,
                is_autocast=True,
                ):
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    total_acc_num = 0
    iter_idx = 1
    total_iters = global_total_iters

    model.train()

    for batch in tqdm_train_set:
        if is_autocast:
            with autocast():
                loss, acc_num = process_one_batch(model, batch, loss_fn)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, acc_num = process_one_batch(model, batch, loss_fn)
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        loss_val = loss.item()
        total_train_loss += loss_val
        total_acc_num += acc_num.item()
        if global_rank == 0:
            if iter_idx % logging_frequency == 0:
                wandb.log({
                "train_loss": loss_val,
                "total_iters": total_iters,
                "epoch": epoch,
                }, step=total_iters)
        tqdm_train_set.set_postfix({str(global_rank)+'_train_acc': total_acc_num / max((iter_idx*batch_size), 1)})
        iter_idx += 1
        total_iters += 1

    num_examples = max((iter_idx-1)*batch_size, 1)
    train_acc = total_acc_num / num_examples
    ave_train_loss = total_train_loss / num_examples
    return train_acc, ave_train_loss, total_iters-1

# do one epoch for eval
def eval_epoch(model,
                eval_set,
                loss_fn,
                batch_size
                ):
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    total_acc_num = 0
    iter_idx = 1
    model.eval()
  
    # Evaluate data for one epoch
    for batch in tqdm_eval_set: 
        with torch.no_grad():
            loss, acc_num = process_one_batch(model, batch, loss_fn)
            total_eval_loss += loss.item()
            total_acc_num += acc_num.item()
        tqdm_eval_set.set_postfix({str(global_rank)+'_eval_acc': total_acc_num / max((iter_idx*batch_size), 1)})
        iter_idx += 1
    
    num_examples = max(((iter_idx-1)*batch_size), 1)
    eval_acc = total_acc_num / num_examples
    ave_eval_loss = total_eval_loss / num_examples
    return eval_acc, ave_eval_loss

def read_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(args):
    config = read_config_from_yaml(args.train_config_path)
    LOAD_FROM_CHECKPOINT = args.load_from_checkpoint
    config['load_from_checkpoint'] = LOAD_FROM_CHECKPOINT

    print(config)

    TRAIN_FOLDERS = config.get("train_folders")
    EVAL_FOLDERS = config.get("eval_folders")

    PRE_WEIGHTS_PATH = config.get("pre_weights_path")

    BASE_DIR = config.get("base_dir", None)
    WEIGHTS_PATH = f'{BASE_DIR}/{config.get("weights_path")}'
    LOGS_PATH = f'{BASE_DIR}/{config.get("logs_path")}'
    CHECKPOINT_PATH = f'{BASE_DIR}/{config.get("checkpoint_path")}'
    DATALOADER_PATH = f'{BASE_DIR}/{config.get("dataloader_path")}'
    
    PATCH_SIZE = config.get("patch_size")
    PATCH_LENGTH = config.get("patch_length")
    BYTE_NUM_LAYERS = config.get("byte_num_layers")
    PATCH_NUM_LAYERS = config.get("patch_num_layers")
    HIDDEN_SIZE = config.get("hidden_size")
    
    NUM_EPOCHS = config.get("num_epochs")
    LEARNING_RATE = config.get("learning_rate")
    BATCH_SIZE = config.get("batch_size")
    PATCH_SAMPLING_BATCH_SIZE = config.get("patch_sampling_batch_size")
    
    
    LOAD_FROM_PRE_CHECKPOINT = config.get("load_from_pre_checkpoint")   
    LOGGING_FREQUENCY = config.get("logging_frequency")
    
    WANDB_CONFIG = config.get("wandb")
    WANDB_PROJ_NAME = WANDB_CONFIG.get("proj_name")
    WANDB_ENTITY = WANDB_CONFIG.get("entity")
    WANDB_MODE = WANDB_CONFIG.get("mode")
    WANDB_NAME = WANDB_CONFIG.get("name", "run")

    batch_size = BATCH_SIZE

    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
    Path(DATALOADER_PATH).mkdir(parents=True, exist_ok=True)

    if global_rank == 0:
        wandb.init(project=WANDB_PROJ_NAME, 
                entity=WANDB_ENTITY, 
                mode=WANDB_MODE,
                dir=BASE_DIR,
                name=WANDB_NAME + f"_{datetime.now().strftime('%Y%m%d_%H%M_%S')}")

        wandb.config.update({
            "TRAIN_FOLDERS": TRAIN_FOLDERS,
            "EVAL_FOLDERS": EVAL_FOLDERS,
            # "PRE_WEIGHTS_PATH": PRE_WEIGHTS_PATH,
            "WEIGHTS_PATH": WEIGHTS_PATH,
            "LOGS_PATH": LOGS_PATH,
            "PATCH_SIZE": PATCH_SIZE,
            "PATCH_LENGTH": PATCH_LENGTH,
            "BYTE_NUM_LAYERS": BYTE_NUM_LAYERS,
            "PATCH_NUM_LAYERS": PATCH_NUM_LAYERS,
            "HIDDEN_SIZE": HIDDEN_SIZE,
            "NUM_EPOCHS": NUM_EPOCHS,
            "LEARNING_RATE": LEARNING_RATE,
            "BATCH_SIZE": BATCH_SIZE,
            "LOAD_FROM_CHECKPOINT": LOAD_FROM_CHECKPOINT,
            "LOAD_FROM_PRE_CHECKPOINT": LOAD_FROM_PRE_CHECKPOINT
            # Add any other configurations you'd like to track
        })            
    # load filenames under train and eval folder
    train_files = list_files_in_directory(TRAIN_FOLDERS)
    eval_files = list_files_in_directory(EVAL_FOLDERS)

    train_set = ByteDataset(train_files, PATCH_SIZE, PATCH_LENGTH)
    eval_set = ByteDataset(eval_files, PATCH_SIZE, PATCH_LENGTH)

    patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, 
                        max_length=PATCH_LENGTH, 
                        max_position_embeddings=PATCH_LENGTH,
                        hidden_size=HIDDEN_SIZE,
                        n_head=HIDDEN_SIZE//64,
                        vocab_size=1)
    model = bGPTForClassification(patch_config, len(train_set.labels), PATCH_SIZE)
    model = model.to(device)

    # print parameter number
    print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,  find_unused_parameters=True)

    scaler = GradScaler()
    is_autocast = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    labels = train_set.labels

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=global_rank)
    eval_sampler = DistributedSampler(eval_set, num_replicas=world_size, rank=global_rank)

    train_set = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_batch, sampler=train_sampler, shuffle = (train_sampler is None))
    eval_set = DataLoader(eval_set, batch_size=BATCH_SIZE, collate_fn=collate_batch, sampler=eval_sampler, shuffle = (train_sampler is None))

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=NUM_EPOCHS * len(train_set) // 10,
        num_training_steps=NUM_EPOCHS * len(train_set),
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    if LOAD_FROM_PRE_CHECKPOINT and os.path.exists(PRE_WEIGHTS_PATH):
        # Load checkpoint to CPU
        checkpoint = torch.load(PRE_WEIGHTS_PATH, map_location='cpu')

        byte_config = GPT2Config(num_hidden_layers=BYTE_NUM_LAYERS, 
                            max_length=PATCH_SIZE+1, 
                            max_position_embeddings=PATCH_SIZE+1,
                            hidden_size=HIDDEN_SIZE,
                            n_head=HIDDEN_SIZE//64,
                            vocab_size=256+1)
        pretrained_model = bGPTLMHeadModel(patch_config, byte_config, PATCH_SIZE, PATCH_SAMPLING_BATCH_SIZE)
        pretrained_model.load_state_dict(checkpoint['model'])

        # Here, model is assumed to be on GPU
        # Load state dict to CPU model first, then move the model to GPU
        if torch.cuda.device_count() > 1:
            # If you have a DataParallel model, you need to load to model.module instead
            cpu_model = deepcopy(model.module)
            cpu_model.patch_level_decoder.load_state_dict(pretrained_model.patch_level_decoder.state_dict())
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            # Load to a CPU clone of the model, then load back
            cpu_model = deepcopy(model)
            cpu_model.patch_level_decoder.load_state_dict(pretrained_model.patch_level_decoder.state_dict())
            model.load_state_dict(cpu_model.state_dict())
        
        try:
            print(f"Successfully Loaded Pretrained Checkpoint at Epoch {checkpoint['epoch']} with Loss {checkpoint['min_eval_loss']}")
        except:
            print(f"Successfully Loaded Pretrained Checkpoint at Epoch {checkpoint['epoch']} with Acc {checkpoint['max_eval_acc']}")

    if LOAD_FROM_CHECKPOINT and os.path.exists(WEIGHTS_PATH):
        # Load checkpoint to CPU
        checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
        # Here, model is assumed to be on GPU
        # Load state dict to CPU model first, then move the model to GPU
        if torch.cuda.device_count() > 1:
            # If you have a DataParallel model, you need to load to model.module instead
            cpu_model = deepcopy(model.module)
            cpu_model.load_state_dict(checkpoint['model'])
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            # Load to a CPU clone of the model, then load back
            cpu_model = deepcopy(model)
            cpu_model.load_state_dict(checkpoint['model'])
            model.load_state_dict(cpu_model.state_dict())

        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_sched'])
        pre_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        max_eval_acc = checkpoint['max_eval_acc']
        labels = checkpoint['labels']
        total_iters = checkpoint['total_iters']
        print("Successfully Loaded Checkpoint from Epoch %d" % pre_epoch)
    
    else:
        pre_epoch = 0
        best_epoch = 0
        max_eval_acc = 0
        total_iters = 0

    for epoch in range(1, NUM_EPOCHS+1-pre_epoch):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        epoch += pre_epoch
        print('-' * 21 + "Epoch " + str(epoch) + '-' * 21)

        train_acc, ave_train_loss, total_iters = train_epoch(model,
                                                                train_set,
                                                                lr_scheduler,
                                                                scaler,
                                                                optimizer,
                                                                loss_fn,
                                                                epoch,
                                                                BATCH_SIZE,
                                                                LOGGING_FREQUENCY,
                                                                total_iters+1,
                                                            )
        eval_acc, ave_eval_loss = eval_epoch(model,
                                            eval_set,
                                            loss_fn,
                                            BATCH_SIZE
                                        )
        if global_rank==0:
            with open(LOGS_PATH,'a') as f:
                f.write("Epoch " + str(epoch) + "\ntrain_acc: " + str(train_acc) + "\neval_acc: " +str(eval_acc) + "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n")
            if eval_acc > max_eval_acc:
                best_epoch = epoch
                max_eval_acc = eval_acc
            # Log and save checkpoint every epoch regardless
            wandb.log({"epoch": epoch,
                        "best_epoch": best_epoch,
                        "total_iters": total_iters,
                        'train_acc': train_acc,
                        'ave_train_loss': ave_train_loss,
                        'eval_acc': eval_acc,
                        'ave_eval_loss': ave_eval_loss,
                        'max_eval_acc': max_eval_acc,
                        }, step=total_iters)
            
            checkpoint = { 
                            'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_sched': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'best_epoch': best_epoch,
                            'train_acc': train_acc,
                            'max_eval_acc': max_eval_acc,
                            "labels": labels,
                            'total_iters': total_iters,
                            }
            
            checkpoint_filepath = f'{CHECKPOINT_PATH}/checkpoint_{epoch}.pth'
            torch.save(checkpoint, checkpoint_filepath)
            torch.save(checkpoint, f'{CHECKPOINT_PATH}/latest.pth')
            # Save latest_checkpoint_info.json about the latest checkpoint info and path.
            with open(f'{CHECKPOINT_PATH}/latest_checkpoint_info.json', "w") as f:
                checkpoint_data = {
                    "latest_checkpoint_path": checkpoint_filepath,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "total_iters": total_iters,
                    'train_acc': train_acc,
                    'ave_train_loss': ave_train_loss,
                    'eval_acc': eval_acc,
                    'ave_eval_loss': ave_eval_loss,
                    'max_eval_acc': max_eval_acc,
                    # "train_loss": minibatch_loss,
                    # "ave_eval_loss": eval_loss if eval_loss else 0,
                }
                json.dump(checkpoint_data, f)

            print(f"Checkpoint saved at {checkpoint_filepath}")
        
        if world_size > 1:
            dist.barrier()

    if global_rank==0:
        print("Best Eval Epoch : "+str(best_epoch))
        print("Max Eval Accuracy : "+str(max_eval_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for bGBT classification")
    parser.add_argument("--train-config-path", type=str, required=True,
                        help="Path to the config YAML file for training run")
    parser.add_argument("--load-from-checkpoint", action='store_true', dest="load_from_checkpoint",
                        help="If ths flag is present, model checkpoint will be loaded. By default without the flag, checkpoint will not be loaded.")
    parser.set_defaults(load_from_checkpoint=False)
    args = parser.parse_args()
    main(args)