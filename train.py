import json
import time
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from trainer.asr.trainer import Trainer

from utils import constant
from utils.data_loaders.data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler
from utils.functions import save_model, load_model, init_transformer_model, init_optimizer
from utils.parallel import DataParallel
from utils.logger import Logger

import sys

if __name__ == '__main__':
    args = constant.args
    logger = Logger("log/" + args.name)
    sys.stdout = logger

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    print("audio_conf", audio_conf)

    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file))).lower()

    # add PAD_CHAR, SOS_CHAR, EOS_CHAR
    labels = constant.PAD_CHAR + constant.SOS_CHAR + constant.EOS_CHAR + labels
    label2id = dict([(labels[i], i) for i in range(len(labels))])
    id2label = dict([(i, labels[i]) for i in range(len(labels))])

    train_data = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, label2id=label2id,
                                    normalize=True, augment=args.augment)
    train_sampler = BucketingSampler(train_data, batch_size=args.batch_size)
    train_loader = AudioDataLoader(
        train_data, num_workers=args.num_workers, batch_sampler=train_sampler)

    valid_data = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, label2id=label2id,
                                    normalize=True, augment=False)
    valid_sampler = BucketingSampler(valid_data, batch_size=args.batch_size)
    valid_loader = AudioDataLoader(valid_data, num_workers=args.num_workers)

    test_data = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest, label2id=label2id,
                                   normalize=True, augment=False)
    test_loader = AudioDataLoader(test_data, num_workers=args.num_workers)

    start_epoch = 0
    metrics = None
    loaded_args = None
    if constant.args.continue_from != "":
        print("Continue from checkpoint:", constant.args.continue_from)
        model, opt, epoch, metrics, loaded_args, label2id, id2label = load_model(
            constant.args.continue_from)
        start_epoch = (epoch-1)  # index starts from zero
        verbose = constant.args.verbose
    else:
        if constant.args.model == "TRFS":
            model = init_transformer_model(constant.args, label2id, id2label)
            opt = init_optimizer(constant.args, model, "noam")
        else:
            print("The model is not supported, check args --h")
    
    loss_type = args.loss

    if constant.USE_CUDA:
        model = model.cuda()

    # Parallelize the batch
    if args.parallel:
        device_ids = args.device_ids
        model = DataParallel(model, device_ids=device_ids)
    else:
        if loaded_args != None:
            if loaded_args.parallel:
                print("unwrap from DataParallel")
                model = model.module

    print(model)
    num_epochs = constant.args.epochs

    trainer = Trainer()
    trainer.train(model, train_loader, train_sampler, valid_loader, opt, loss_type, start_epoch, num_epochs, label2id, id2label, metrics, logger)
