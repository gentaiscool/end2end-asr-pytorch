# End-to-end speech recognition on Pytorch

Implementation of Transformer ASR

Train
```
CUDA_VISIBLE_DEVICES=2 python3 train.py --train-manifest data/manifests/train_manifest.csv --val-manifest data/manifests/val_manifest.csv --test-manifest data/manifests/test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/label.json --lr 1e-4 --name model_name --save-folder save/ --save-every 1 --emb_cnn --shuffle
```

Multi-train
```
CUDA_VISIBLE_DEVICES=0,1 python3 multi_train.py --train-manifest-list data/manifests/train_manifest1.csv data/manifests/train_manifest2.csv --val-manifest-list data/manifests/dev_manifest1.csv data/manifests/val_manifest2.csv --test-manifest data/manifests/test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/label.json --lr 1e-4 --name model_name --save-folder save/ --emb_cnn --parallel --device-ids 0 1 --shuffle
```