# End-to-end speech recognition on Pytorch

### Highlights
- implemented Transformer model (from <a href="https://arxiv.org/abs/1804.10752">Syllable-Based Sequence-to-Sequence Speech Recognition with the Transformer in Mandarin Chinese<a/> with some modifications on the encoder input (additional CNN layer)
- supports training on multiple corpora (with any languages)
- supports batch parallelization on multi GPUs

### Requirements
- Python 3.5
- Install Pytorch 1.0 (https://pytorch.org/)
- Install torchaudio (https://github.com/pytorch/audio)
- run ``bash requirement.txt``

## Data
### Librispeech
This is the script provided by <a href="https://github.com/SeanNaren/deepspeech.pytorch">Sean Naren</a>. You can specify which librispeech files you want.

```
python3 data/librispeech.py --target-dir LibriSpeech_dataset/ --sample-rate 16000 --files-to-use train-clean-100.tar.gz,train-clean-360.tar.gz,train-other-500.tar.gz,dev-clean.tar.gz,dev-other.tar.gz,test-clean.tar.gz,test-other.tar.gz --min-duration 1 --max-duration 15
```

### Custom Dataset
#### Manifest file
To use your own dataset, you must create a CSV manifest file using the following format:

```
/path/to/audio.wav,/path/to/text.txt
/path/to/audio2.wav,/path/to/text2.txt
...
```
Each line contains the path to the audio file and transcript file separated by a comma.

#### Label file
You need to specify all characters in the corpus by using the following JSON format:

```
[ 
  "_",
  "'",
  "A",
  ...,
  "Z",
  " "
]
```

### Training
```
usage: train.py [-h] [--train-manifest] [--val-manifest] [--test-manifest] [--cuda] [--verbose] [--batch-size] [--labels-path] [--lr] [--name] [--save-folder] [--save-every] [--emb_cnn] [--emb_trg_sharing] [--shuffle] [--sample_rate] [--label-smoothing] [--window-size] [--window-stride] [--window] [--epochs]  [--src-max-len] [--tgt-max-len] [--warmup] [--momentum] [--lr-anneal] [--num-layers] [--num-heads] [--dim-model] [--dim-key] [--dim-value] [--dim-input] [--dim-inner] [--dim-emb] [--shuffle]
```
#### Parameters
```
- emb_cnn: add 2D convolutional layer in the input
- cuda: train on GPU
- shuffle: randomly shuffle every batch
```

#### Example
```
python train.py --train-manifest data/libri_train_manifest.csv --val-manifest libri_val_manifest.csv --test-manifest libri_test_clean_manifest.csv --labels-path data/labels/labels.json --emb_cnn --shuffle
```

Use ``python train.py --help`` for more parameters and options.

### Multi-GPU Training
```
usage: multi_train.py [-h] [--train-manifest-list] [--val-manifest-list] [--test-manifest] [--cuda] [--verbose] [--batch-size] [--labels-path] [--lr] [--name] [--save-folder] [--save-every] [--emb_cnn] [--emb_trg_sharing] [--shuffle] [--sample_rate] [--label-smoothing] [--window-size] [--window-stride] [--window] [--epochs]  [--src-max-len] [--tgt-max-len] [--warmup] [--momentum] [--lr-anneal] [--num-layers] [--num-heads] [--dim-model] [--dim-key] [--dim-value] [--dim-input] [--dim-inner] [--dim-emb] [--shuffle] [--parallel] [--device-ids]
```
#### Parameters
```
- emb_cnn: add 2D convolutional layer in the input
- cuda: train on GPU
- shuffle: randomly shuffle every batch
- parallel: split batches to GPUs
- device-ids: GPU ids
```

#### Example
```
python multi_train.py --train-manifest-list data/libri_train_manifest.csv --val-manifest-list libri_val_manifest.csv --test-manifest libri_test_clean_manifest.csv --labels-path data/labels/labels.json --emb_cnn --shuffle --parallel --device-ids 0 1
```

Use ``python multi_train.py --help`` for more parameters and options.

## Todo
- [x] parallelization
- [ ] better documentation
- [ ] add more models (LAS, CTC)
- [ ] experiment results
