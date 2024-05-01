## Training Audio Mnist


### Step1:
Run:
```
cd {path/to}/byte_models/scripts/audio-mnist
git clone git@github.com:soerenab/AudioMNIST.git
```

#### step 1-2(optional):
Your can make a directory called `audio-mnist` in your scratch folder and copy the `data` direcory of audio-mnist repo  to your scratch:
```
mkdir -p /{your/scratch/folder}/audio-mnist
cd path/to/byte_models/scripts/audio-mnist/AudioMNIST
cp -rf data /your/scratch/folder
```

#### step 1-3:
make other necessary directories:
```
cd {path/to}/byte_models/scripts/audio-mnist
mkdir -p exp
cd exp
mkdir -p chkp
```

### step2:
split the data: **(change the path inside if required)**
```
cd {path/to}/byte_models/scripts/audio-mnist
# python simple_split.py --root_dir {/audio-mnist/data} --output_path {train/test/val/base/directory}
```
These scripts split the data into `train`, `test` and `val`.

### Step3:
compress the audio: **(change the path inside if required)**
```
{path/to}/byte_models/scripts/audio-mnist
# python audio_compressor.py --base_dir {your/audio-mnist-data}
```

### Step4:
Add your `train` and `val` path to `config_og` file in `TRAIN_FOLDERS` and `EVAL_FOLDERS` respectively, if not exited:

### Step5:
refer to this configuration part, for changing your configs:

<a>https://github.com/CERC-AAI/byte_models/blob/main/bgpt/README.md#configuration</a>

**NOTE**: Addition to that, chnage your `wandb` info in the file.

### Step6:

#### step6-1:
Save your `wandb api key` in a file called `.wanddb_config` and place it in `/your/home/` 

#### step6-2:
Change your sbatch argument in `train.sh` if needed.

#### step 6-3:
correct your env name (if not `bgpt`) in the `train.sh`

#### step6-4:
Run the following command to start training:
```
cd {path/to}/byte_models/scripts/audio-mnist
sbatch train.sh
```