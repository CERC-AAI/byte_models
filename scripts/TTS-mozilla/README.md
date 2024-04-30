## Training Mozila 

unfortunately, Mozilla common voice **can not** be downloaded via script.

### Step1:
You can go the following link and download `Common Voice Delta Segment 17.0` for small verion or `Common Voice Corpus 17.0` for large version of dataset:

<a>https://commonvoice.mozilla.org/en/datasets</a>

enter your `email` and download.

### Step2:

once you downloade the data, transfer them to your desired directory.

```
cp /downloaded/path desired/path
```

After that, run the followings: 
```
cd path/to/scripts/TTS-Mozilla
mkdir exp
cd exp
mkdir chkp
```

for **generative** training, run:**(change the path inside if required)**
```
python audio_segment_without_label.py
```
for **classification** training run:**(change the path inside if required)**
```
python audio_compressor.py
```

### step3:
split the data: **(change the path inside if required)**
```
python split/data.py
```
These scripts will chunk the data to sizes<=8k, split them into `train`, `test` and `val`.

**NOTE:** Make sure you change the paths that you want to store your splits in.

### Step4:
Add your `train` and `val` path to `config_og` file in `TRAIN_FOLDERS` and `EVAL_FOLDERS` respectively, if not exited:

### Step5:
refer to this configuration part, for changing your logs

<a>https://github.com/CERC-AAI/byte_models/blob/main/bgpt/README.md#configuration</a>

**NOTE**: Addition to that, chnage your `wandb` info in the file.



### Step6:

#### step6-1:
Save your `wandb api key` in a file called `.wanddb_config` and place it in `/your/home/` 

#### step6-2:
Change your sbatch argument in `train.sh` if needed.

#### step 6-3:
correct your env name (if not `bgpt3`) in the `train.sh`

#### step6-4:
Run the following command to start training:
```
cd path/to/scripts/TTS-Mozilla
sbatch train.sh
```