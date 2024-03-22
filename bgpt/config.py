# Configuration for generative modelling and classification
TRAIN_FOLDERS = [
                # "wikipedia/train",
                ""  
                "/lustre/orion/csc590/scratch/george-adams/bgpt/wikipedia/train_text3", 
                "/lustre/orion/csc590/scratch/george-adams/bgpt/ag_news/train_text2", 
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
                "/lustre/orion/csc590/scratch/george-adams/bgpt/wikipedia/test",  
                "/lustre/orion/csc590/scratch/george-adams/bgpt/ag_news/test", 
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
PRE_WEIGHTS_PATH = "/localdisks/rogeralexis/george_files/byte_models/bgpt/models/bgpt/weights-image.pth"                           # Path to pre-trained weights
WEIGHTS_PATH = "/lustre/orion/csc590/scratch/george-adams/bgpt/models/bgpt/16-nodes-test1234.pth"                        # Path to save weights
LOGS_PATH = "/lustre/orion/csc590/scratch/george-adams/bgpt/legit_runs/simple_math_test1.txt"                              # Path to save logs

# Configuration for the model
PATCH_SIZE = 16                                                 # Patch Size
PATCH_LENGTH = 1024                                              # Patch Length
BYTE_NUM_LAYERS = 3                                             # Number of layers in the decoder
PATCH_NUM_LAYERS = 12                                           # Number of layers in the encoder
HIDDEN_SIZE = 768                                               # Hidden Size

# Configuration for the training
NUM_EPOCHS = 32                                                 # Number of epochs to train for (if early stopping doesn't intervene)
LEARNING_RATE = 1e-4                                            # Learning rate for the optimizer
BATCH_SIZE = 16                                                  # Batch size for training
ACCUMULATION_STEPS = 1                                          # Accumulation steps to simulate large batch size
PATCH_SAMPLING_BATCH_SIZE = 0                                   # Batch size for patch during training, 0 for full conaudio
LOAD_FROM_CHECKPOINT = False                                    # Whether to load weights from a checkpoint
LOAD_FROM_PRE_CHECKPOINT = False                                 # Whether to load pre-trained weights from a checkpoint

# Configuration for inference
INFERENCE_WEIGHTS_PATH = "/lustre/orion/csc590/scratch/george-adams/bgpt/models/bgpt/16-nodes-test9.pth"               # Path to weights for inference
INPUT_EXT = "txt"                                               # Extension of input files, used for conversion
TARGET_EXT = "txt"                                              # Extension of target files
INPUT_FOLDER = "/lustre/orion/csc590/scratch/george-adams/bgpt/input_text"                                          # Folder containing input files
OUTPUT_FOLDER = "/lustre/orion/csc590/scratch/george-adams/bgpt/output_text"                                        # Folder to save output files
MODE = "generate"                                                # Mode of inference (convert or generate)
NUM_SAMPLES = 5                                               # Number of samples to generate (only for generate mode)
TOP_K = 0                                                       # Top k for sampling
TOP_P = 1.                                                      # Top p for sampling
TEMPERATURE = 1                                                 # Temperature for sampling
