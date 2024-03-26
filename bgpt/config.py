# Configuration for generative modelling and classification
TRAIN_FOLDERS = [
                # "wikipedia/train",
                # "/lustre/orion/csc590/scratch/george-adams/bgpt/data2/math-adder/3-digit/train",
                "/lustre/orion/csc590/scratch/george-adams/data/wikipedia/train_text3",
                "/lustre/orion/csc590/scratch/george-adams/data/ag_news/train_text2",
                # "symbolic-instruction-tuning/train",
                # "imagenet32/train",
                # "cifar/train",
                # "librispeech8K/train",
                # "speech_commands8K/train",
                # "irishman-abc/train",
                # "irishman-mid/train",
                # "cpu_states/train",
                 ]     # Folder containing training data
EVAL_FOLDERS = [
                # "/lustre/orion/csc590/scratch/george-adams/bgpt/wikipedia/test",
                # "/lustre/orion/csc590/scratch/george-adams/bgpt/ag_news/test",
                # "/lustre/orion/csc590/scratch/george-adams/bgpt/data2/math-adder/3-digit/test",
                # "symbolic-instruction-tuning/test",
                # "imagenet32/test",
                # "cifar/test",
                # "librispeech8K/test",
                # "speech_commands8K/test",
                # "irishman-abc/test",
                # "irishman-mid/test",
                # "cpu_states/test",
                ]                                               # Folder containing evaluation data

# Configuration for the paths
base_path = "/lustre/orion/csc590/scratch/george-adams/bgpt_folders/bgpt_110m-wikipedia-ag-news"  # Base path for the project
WEIGHTS_PATH = f"{base_path}/16-nodes-scratch.pth"                        # Path to save weights
LOGS_PATH = f"{base_path}/log.txt"                              # Path to save logs
CHECKPOINT_PATH = f"{base_path}/checkpoints"                                          # Path to save checkpoints
DATALOADER_PATH = f"{base_path}/dataloaders"                                           # Path to save dataloaders


# Config 110M
PATCH_SIZE = 16                                                 # Patch Size
PATCH_LENGTH = 1024                                              # Patch Length
BYTE_NUM_LAYERS = 3                                             # Number of layers in the decoder
PATCH_NUM_LAYERS = 12                                           # Number of layers in the encoder
HIDDEN_SIZE = 768                                               # Hidden Size

# Config 410M
# PATCH_SIZE = 16                                                 # Patch Size
# PATCH_LENGTH = 1024                                              # Patch Length
# BYTE_NUM_LAYERS = 5                                             # Number of layers in the decoder
# PATCH_NUM_LAYERS = 19                                           # Number of layers in the encoder
# HIDDEN_SIZE = 1024                                               # Hidden Size

# Configuration for the training
NUM_EPOCHS = 32                                                 # Number of epochs to train for (if early stopping doesn't intervene)
LEARNING_RATE = 1e-4                                            # Learning rate for the optimizer
BATCH_SIZE = 16                                                  # Batch size for training
ACCUMULATION_STEPS = 1                                          # Accumulation steps to simulate large batch size
PATCH_SAMPLING_BATCH_SIZE = 0                                   # Batch size for patch during training, 0 for full conaudio
LOAD_FROM_CHECKPOINT = False                                    # Whether to load weights from a checkpoint

CHECKPOINT_FREQUENCY = 20

