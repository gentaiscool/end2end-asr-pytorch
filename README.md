# End-to-end speech recognition on Pytorch

Implementation of end-to-end ASR
- Transformer model (from <a href="https://arxiv.org/abs/1804.10752">Syllable-Based Sequence-to-Sequence Speech Recognition with the Transformer in Mandarin Chinese<a/> with some modifications on encoder input (additional CNN layer)
- support training on multiple corpora (with any language)
- support batch parallelization on multi GPUs

Requirements:
- Pytorch 1.0
- tqdm
- torchaudio (https://github.com/pytorch/audio)

train.py
```
CUDA_VISIBLE_DEVICES=2 python3 train.py --train-manifest data/manifests/train_manifest.csv --val-manifest data/manifests/val_manifest.csv --test-manifest data/manifests/test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/label.json --lr 1e-4 --name model_name --save-folder save/ --save-every 1 --emb_cnn --shuffle
```

multi-train.py
```
CUDA_VISIBLE_DEVICES=0,1 python3 multi_train.py --train-manifest-list data/manifests/train_manifest1.csv data/manifests/train_manifest2.csv --val-manifest-list data/manifests/dev_manifest1.csv data/manifests/val_manifest2.csv --test-manifest data/manifests/test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/label.json --lr 1e-4 --name model_name --save-folder save/ --emb_cnn --parallel --device-ids 0 1 --shuffle
```
Todo
- [x] parallelization
- [ ] better documentation
- [ ] add more models (LAS, CTC)
