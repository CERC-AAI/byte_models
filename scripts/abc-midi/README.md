#### Configuration
clone this repo for abc->midi conversion. Same repository was mentiond in the paper for convesion.
```
    cd path/to/scripts/abc-midi
    git clone git@github.com:xlvector/abcmidi.git
    cd abcmidi
    cmake
```

**NOTE** change the dir for save accordingly.
### Download irishman dataset
```
    cd path/to/scripts/abc-midi
    python prep_abc_data.py sander-wood/irishman ../../data/hf_cache --split train -o ../../data/abc_data/train
    python prep_abc_data.py sander-wood/irishman ../../data/hf_cache --split validation -o ../../data/abc_data/val
```
### abc to midi file conversion
```
    cd path/to/scripts/abc-midi
    python abc2midi.py ../../data/abc_data/train -o ../../data/midi_data/train
    python abc2midi.py ../../data/abc_data/val -o ../../data/midi_data/val
```
