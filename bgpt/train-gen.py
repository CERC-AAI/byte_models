import os
import glob
import time
import torch
import random
import numpy as np
import yaml
import argparse

from utils import *
# from config import *
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2Config, get_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import wandb

from torch.utils.data import Dataset, DataLoader

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

# print(f'GLOBAL RANK: {global_rank}')
# print(dist.get_rank())


# Set random seed
seed = 0 + global_rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def find_most_recent_file(directory, pattern="*.pth"):
    file_paths = glob.glob(os.path.join(directory, pattern))
    if file_paths:
        return max(file_paths, key=os.path.getmtime)
    return None


def collate_batch(input_batches):
    input_patches, input_masks = zip(*input_batches)
    input_patches = torch.nn.utils.rnn.pad_sequence(input_patches, batch_first=True, padding_value=256)
    input_masks = torch.nn.utils.rnn.pad_sequence(input_masks, batch_first=True, padding_value=0)

    return input_patches.to(device), input_masks.to(device)


def split_into_minibatches(input_patches, input_masks, minibatch_size):
    minibatches = []
    for start_idx in range(0, len(input_patches), minibatch_size):
        end_idx = start_idx + minibatch_size
        minibatch_patches = input_patches[start_idx:end_idx]
        minibatch_masks = input_masks[start_idx:end_idx]
        minibatches.append((minibatch_patches, minibatch_masks))
    return minibatches


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
               patch_length):
    ext = filename.split('.')[-1]
    ext = bytearray(ext, 'utf-8')
    ext = [byte for byte in ext][:patch_size]

    with open(filename, 'rb') as f:
        file_bytes = f.read()

    bytes = []
    for byte in file_bytes:
        bytes.append(byte)

    if len(bytes) % patch_size != 0:
        bytes = bytes + [256] * (patch_size - len(bytes) % patch_size)

    bos_patch = ext + [256] * (patch_size - len(ext))
    bytes = bos_patch + bytes + [256] * patch_size
    bytes = bytes[:patch_length * patch_size]
    masks = [1] * (len(bytes) // patch_size)

    return bytes, masks


class ByteDataset(Dataset):
    def __init__(self, filenames, patch_size, patch_length):
        self.filenames = filenames
        self.patch_size = patch_size
        self.patch_length = patch_length

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        file_bytes, file_masks = read_bytes(filename, self.patch_size, self.patch_length)

        file_bytes = torch.tensor(file_bytes, dtype=torch.long)
        file_masks = torch.tensor(file_masks, dtype=torch.long)

        return file_bytes, file_masks


# call model with a batch of input
def process_one_batch(batch, 
                      model, 
                      world_size
                      ):
    input_patches, input_masks = batch
    loss = model(input_patches, input_masks).loss

    # Reduce the loss on GPU 0
    if world_size > 1:
        loss = loss.unsqueeze(0)
        dist.reduce(loss, dst=0)
        loss = loss / world_size
        dist.broadcast(loss, src=0)

    return loss


# do one epoch for training
def train_epoch(model,
                train_set, 
                lr_scheduler, 
                scaler,
                optimizer,
                epoch, 
                best_epoch, 
                min_eval_loss, 
                batch_size, 
                accumulation_steps, 
                checkpoint_frequency, 
                checkpoint_path):
    
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    iter_idx = 1
    checkpoint_iters = 0
    total_iters = 0
    model.train()

    for batch in tqdm_train_set:
        minibatches = split_into_minibatches(batch[0], batch[1], batch_size // accumulation_steps)
        for minibatch in minibatches:
            with autocast():
                loss = process_one_batch(minibatch, model, world_size) / accumulation_steps
            scaler.scale(loss).backward()
            total_train_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        tqdm_train_set.set_postfix({str(global_rank) + '_train_loss': total_train_loss / iter_idx})

        if iter_idx == checkpoint_frequency:
            checkpoint = {
                'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_epoch': best_epoch,
                'min_eval_loss': min_eval_loss
            }

        if checkpoint_iters == checkpoint_frequency:
            torch.save(checkpoint, f'{checkpoint_path}/checkpoint{total_iters}.pth')
            # torch.save(dataloader.state_dict(), f'{DATALOADER_PATH}/dataloader{total_iters}.pth') # THIS doesn't work
            checkpoint_iters = 0

        checkpoint_iters += 1
        total_iters += 1
        iter_idx += 1

    return total_train_loss / (iter_idx - 1)


# do one epoch for eval
def eval_epoch(model,
               eval_set, 
               batch_size, 
               accumulation_steps
               ):
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    iter_idx = 1
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm_eval_set:
        minibatches = split_into_minibatches(batch[0], batch[1], batch_size // accumulation_steps)
        for minibatch in minibatches:
            with torch.no_grad():
                loss = process_one_batch(minibatch) / accumulation_steps
            total_eval_loss += loss.item()
        tqdm_eval_set.set_postfix({str(global_rank) + '_eval_loss': total_eval_loss / iter_idx})
        iter_idx += 1
    # return total_eval_loss / (iter_idx-1)

    return 0

def read_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(args):
    config = read_config_from_yaml(args.train_config_path)
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
    ACCUMULATION_STEPS = config.get("accumulation_steps")
    PATCH_SAMPLING_BATCH_SIZE = config.get("patch_sampling_batch_size")
    LOAD_FROM_CHECKPOINT = config.get("load_from_checkpoint")
    LOAD_FROM_PRE_CHECKPOINT = config.get("load_from_pre_checkpoint")
    CHECKPOINT_FREQUENCY = config.get("checkpoint_frequency")
    WANDB_CONFIG = config.get("wandb")
    WANDB_PROJ_NAME = WANDB_CONFIG.get("proj_name")
    WANDB_ENTITY = WANDB_CONFIG.get("entity")
    WANDB_MODE = WANDB_CONFIG.get("mode")

    FIRST_LAUNCH = config.get("first_launch")

    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
    Path(DATALOADER_PATH).mkdir(parents=True, exist_ok=True)

    wandb.init(project=WANDB_PROJ_NAME, entity=WANDB_ENTITY, mode=WANDB_MODE)

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
        "ACCUMULATION_STEPS": ACCUMULATION_STEPS,
        "LOAD_FROM_CHECKPOINT": LOAD_FROM_CHECKPOINT,
        "LOAD_FROM_PRE_CHECKPOINT": LOAD_FROM_PRE_CHECKPOINT
        # Add any other configurations you'd like to track
    })

    batch_size = BATCH_SIZE

    patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS,
                            max_length=PATCH_LENGTH,
                            max_position_embeddings=PATCH_LENGTH,
                            hidden_size=HIDDEN_SIZE,
                            n_head=HIDDEN_SIZE // 64,
                            vocab_size=1)
    byte_config = GPT2Config(num_hidden_layers=BYTE_NUM_LAYERS,
                            max_length=PATCH_SIZE + 1,
                            max_position_embeddings=PATCH_SIZE + 1,
                            hidden_size=HIDDEN_SIZE,
                            n_head=HIDDEN_SIZE // 64,
                            vocab_size=256 + 1)
    model = bGPTLMHeadModel(patch_config, byte_config)
    model = model.to(device)

    # print parameter number
    print("Parameter Number: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    scaler = GradScaler()
    is_autocast = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # load filenames under train and eval folder
    train_files = list_files_in_directory(TRAIN_FOLDERS)
    eval_files = list_files_in_directory(EVAL_FOLDERS)

    train_batch_nums = int(len(train_files) / batch_size)
    eval_batch_nums = int(len(eval_files) / batch_size)

    random.shuffle(train_files)
    random.shuffle(eval_files)

    train_files = train_files[:train_batch_nums * batch_size]
    print(f"Number of training files: {len(train_files)}")
    eval_files = eval_files[:eval_batch_nums * batch_size]

    train_dataset = ByteDataset(train_files, PATCH_SIZE, PATCH_LENGTH)
    eval_dataset = ByteDataset(eval_files, PATCH_SIZE, PATCH_LENGTH)

    # Initialize DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=local_rank)

    # Load the most recent DataLoader state (if applicable)
    # if args.first_launch != "True":
    if not FIRST_LAUNCH:
        # For train_set, if you're saving states of the sampler, load it here
        most_recent_train_dataloader_state = find_most_recent_file(DATALOADER_PATH, pattern="train_dataloader_*.pth")
        if most_recent_train_dataloader_state:
            # Assuming you have a mechanism to load the sampler's state
            train_sampler_state = torch.load(most_recent_train_dataloader_state, map_location='cpu')
            train_sampler.load_state_dict(train_sampler_state)
            print(f"Loaded train sampler state from {most_recent_train_dataloader_state}")

        # Similar logic could apply to eval_set if you have a stateful eval_sampler
        most_recent_eval_dataloader_state = find_most_recent_file(DATALOADER_PATH, pattern="eval_dataloader_*.pth")
        if most_recent_eval_dataloader_state:
            # Load eval_sampler state if necessary
            pass

    # Initialize DataLoaders with potentially state-restored samplers
    train_set = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch, sampler=train_sampler,
                           shuffle=False)  # shuffle is False when using a sampler
    eval_set = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_batch, sampler=eval_sampler,
                          shuffle=False)

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

        print(
            f"Successfully Loaded Pretrained Checkpoint at Epoch {checkpoint['epoch']} with Loss {checkpoint['min_eval_loss']}")

    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = 100

    if LOAD_FROM_CHECKPOINT and os.path.exists(WEIGHTS_PATH):
        # Load checkpoint to CPU
        # TODO: Missing function. Is this supposed to be find_most_recent_file?
        # most_recent_checkpoint = find_most_recent_checkpoint(CHECKPOINT_PATH)
        most_recent_checkpoint = find_most_recent_file(CHECKPOINT_PATH, pattern="checkpoint*.pth")
        if most_recent_checkpoint is not None:
            WEIGHTS_PATH = most_recent_checkpoint
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
        min_eval_loss = checkpoint['min_eval_loss']
        print("Successfully Loaded Checkpoint from Epoch %d" % pre_epoch)
        checkpoint = None

    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = 100

    for epoch in range(1 + pre_epoch, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        print('-' * 21 + "Epoch " + str(epoch) + '-' * 21)

        train_loss = train_epoch(model, 
                                 train_set, 
                                 lr_scheduler, 
                                 scaler,
                                 optimizer,
                                 epoch, 
                                 best_epoch, 
                                 min_eval_loss, 
                                 BATCH_SIZE, 
                                 ACCUMULATION_STEPS, 
                                 CHECKPOINT_FREQUENCY, 
                                 CHECKPOINT_PATH
                                 )
        
        eval_loss = eval_epoch(model,
                               eval_set, 
                               BATCH_SIZE, 
                               ACCUMULATION_STEPS
                               )

        if global_rank == 0:
            with open(LOGS_PATH, 'a') as f:
                f.write("Epoch " + str(epoch) + "\ntrain_loss: " + str(train_loss) + "\neval_loss: " + str(
                    eval_loss) + "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n")
            wandb.log({
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "epoch": epoch
            })
            if eval_loss < min_eval_loss:
                best_epoch = epoch
                min_eval_loss = eval_loss
                checkpoint = {
                    'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'min_eval_loss': min_eval_loss
                }
                torch.save(checkpoint, WEIGHTS_PATH)
                # torch.save(dataloader.state_dict(), checkpoint_path)

        if world_size > 1:
            dist.barrier()

    if global_rank == 0:
        print("Best Eval Epoch : " + str(best_epoch))
        print("Min Eval Loss : " + str(min_eval_loss))



# train and eval
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for bGBT")
    parser.add_argument("--train-config-path", type=str, required=True, help="Path to the config YAML file for training run")
    args = parser.parse_args()

    main(args)

