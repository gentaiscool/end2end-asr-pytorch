import json
import time
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from trainer.asr.multi_trainer import MultiTrainer

from utils import constant
from utils.multi_data_loader import MultiSpectrogramDataset, MultiAudioDataLoader, MultiBucketingSampler
from utils.data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler
from utils.functions import save_model, load_model, init_transformer_model, init_deepspeech_model, init_las_model, init_optimizer
from utils.parallel import DataParallel

from utils.logger import Logger

import sys

if __name__ == '__main__':
    args = constant.args
    sys.stdout = Logger("log/" + args.name)

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    print("audio_conf", audio_conf)

    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    # add PAD_CHAR, SOS_CHAR, EOS_CHAR
    labels = constant.PAD_CHAR + constant.SOS_CHAR + constant.EOS_CHAR + labels
    label2id = dict([(labels[i], i) for i in range(len(labels))])
    id2label = dict([(i, labels[i]) for i in range(len(labels))])

    train_manifest_list = args.train_manifest_list
    val_manifest_list = args.val_manifest_list
    test_manifest_list = args.test_manifest_list

    train_data = MultiSpectrogramDataset(audio_conf=audio_conf, manifest_filepath_list=train_manifest_list, label2id=label2id,
                                        normalize=True, augment=args.augment)
    train_sampler = MultiBucketingSampler(train_data, batch_size=args.batch_size)
    train_loader = MultiAudioDataLoader(
        train_data, num_workers=args.num_workers, batch_sampler=train_sampler)

    valid_loaders = []

    for i in range(len(val_manifest_list)):
        val_manifest = val_manifest_list[i]

        valid_data = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=val_manifest, label2id=label2id,
                                        normalize=True, augment=False)
        valid_sampler = BucketingSampler(valid_data, batch_size=args.batch_size)
        valid_loader = AudioDataLoader(valid_data, num_workers=args.num_workers, batch_sampler=valid_sampler)
        valid_loaders.append(valid_loader)

    start_epoch = 0
    metrics = None
    if constant.args.continue_from != "":
        print("Continue from checkpoint:", constant.args.continue_from)
        model, opt, epoch, metrics, loaded_args, label2id, id2label = load_model(
            constant.args.continue_from)
        start_epoch = (epoch-1)  # index starts from zero
        verbose = constant.args.verbose
    else:
        if constant.args.model == "TRFS":
            model = init_transformer_model(constant.args, label2id, id2label)
        elif constant.args.model == "DEEPSPEECH":
            model = init_deepspeech_model(constant.args, label2id, id2label)
        elif constant.args.model == "LAS":
            model = init_las_model(constant.args, label2id, id2label)
        else:
            print("The model is not supported, check args --h")
        opt = init_optimizer(constant.args, model)

    if constant.USE_CUDA:
        print("use cuda")
        model = model.cuda()

    # print(model)
    num_epochs = constant.args.epochs

    trainer = MultiTrainer()
    trainer.train(model, train_loader, train_sampler, valid_loaders, opt, loss_type, start_epoch, num_epochs, label2id, id2label, metrics)
