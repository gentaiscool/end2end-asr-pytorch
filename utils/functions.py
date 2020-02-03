import torch
import os
import math
import torch.nn as nn

from models.asr.transformer import Transformer, Encoder, Decoder
from utils.optimizer import NoamOpt, AnnealingOpt
from utils import constant


def save_model(model, epoch, opt, metrics, label2id, id2label, best_model=False):
    """
    Saving model, TODO adding history
    """
    if best_model:
        save_path = "{}/{}/best_model.th".format(
            constant.args.save_folder, constant.args.name)
    else:
        save_path = "{}/{}/epoch_{}.th".format(constant.args.save_folder,
                                               constant.args.name, epoch)

    if not os.path.exists(constant.args.save_folder + "/" + constant.args.name):
        os.makedirs(constant.args.save_folder + "/" + constant.args.name)

    print("SAVE MODEL to", save_path)
    if constant.args.loss == "ce":
        args = {
            'label2id': label2id,
            'id2label': id2label,
            'args': constant.args,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.optimizer.state_dict(),
            'optimizer_params': {
                '_step': opt._step,
                '_rate': opt._rate,
                'warmup': opt.warmup,
                'factor': opt.factor,
                'model_size': opt.model_size
            },
            'metrics': metrics
        }
    elif constant.args.loss == "ctc":
        args = {
            'label2id': label2id,
            'id2label': id2label,
            'args': constant.args,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.optimizer.state_dict(),
            'optimizer_params': {
                'lr': opt.lr,
                'lr_anneal': opt.lr_anneal
            },
            'metrics': metrics
        }
    else:
        print("Loss is not defined")
    torch.save(args, save_path)


def load_model(load_path):
    """
    Loading model
    args:
        load_path: string
    """
    checkpoint = torch.load(load_path)

    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    if 'args' in checkpoint:
        args = checkpoint['args']

    label2id = checkpoint['label2id']
    id2label = checkpoint['id2label']

    model = init_transformer_model(args, label2id, id2label)
    model.load_state_dict(checkpoint['model_state_dict'])
    if args.cuda:
        model = model.cuda()

    opt = init_optimizer(args, model)
    if opt is not None:
        opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if constant.args.loss == "ce":
            opt._step = checkpoint['optimizer_params']['_step']
            opt._rate = checkpoint['optimizer_params']['_rate']
            opt.warmup = checkpoint['optimizer_params']['warmup']
            opt.factor = checkpoint['optimizer_params']['factor']
            opt.model_size = checkpoint['optimizer_params']['model_size']
        elif constant.args.loss == "ctc":
            opt.lr = checkpoint['optimizer_params']['lr']
            opt.lr_anneal = checkpoint['optimizer_params']['lr_anneal']
        else:
            print("Need to define loss type")

    return model, opt, epoch, metrics, args, label2id, id2label


def init_optimizer(args, model, opt_type="noam"):
    dim_input = args.dim_input
    warmup = args.warmup
    lr = args.lr

    if opt_type == "noam":
        opt = NoamOpt(dim_input, args.k_lr, warmup, torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9), min_lr=args.min_lr)
    elif opt_type == "sgd":
        opt = AnnealingOpt(lr, args.lr_anneal, torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, nesterov=True))
    else:
        opt = None
        print("Optimizer is not defined")

    return opt

def init_transformer_model(args, label2id, id2label):
    """
    Initiate a new transformer object
    """
    if args.feat_extractor == 'emb_cnn':
        hidden_size = int(math.floor(
            (args.sample_rate * args.window_size) / 2) + 1)
        hidden_size = int(math.floor(hidden_size - 41) / 2 + 1)
        hidden_size = int(math.floor(hidden_size - 21) / 2 + 1)
        hidden_size *= 32
        args.dim_input = hidden_size
    elif args.feat_extractor == 'vgg_cnn':
        hidden_size = int(math.floor((args.sample_rate * args.window_size) / 2) + 1) # 161
        hidden_size = int(math.floor(int(math.floor(hidden_size)/2)/2)) * 128 # divide by 2 for maxpooling
        args.dim_input = hidden_size
    else:
        print("the model is initialized without feature extractor")

    num_layers = args.num_layers
    num_heads = args.num_heads
    dim_model = args.dim_model
    dim_key = args.dim_key
    dim_value = args.dim_value
    dim_input = args.dim_input
    dim_inner = args.dim_inner
    dim_emb = args.dim_emb
    src_max_len = args.src_max_len
    tgt_max_len = args.tgt_max_len
    dropout = args.dropout
    emb_trg_sharing = args.emb_trg_sharing
    feat_extractor = args.feat_extractor

    encoder = Encoder(num_layers, num_heads=num_heads, dim_model=dim_model, dim_key=dim_key,
                      dim_value=dim_value, dim_input=dim_input, dim_inner=dim_inner, src_max_length=src_max_len, dropout=dropout)
    decoder = Decoder(id2label, num_src_vocab=len(label2id), num_trg_vocab=len(label2id), num_layers=num_layers, num_heads=num_heads,
                      dim_emb=dim_emb, dim_model=dim_model, dim_inner=dim_inner, dim_key=dim_key, dim_value=dim_value, trg_max_length=tgt_max_len, dropout=dropout, emb_trg_sharing=emb_trg_sharing)
    model = Transformer(encoder, decoder, feat_extractor=feat_extractor)

    if args.parallel:
        device_ids = args.device_ids
        if constant.args.device_ids:
            print("load with device_ids", constant.args.device_ids)
            model = nn.DataParallel(model, device_ids=constant.args.device_ids)
        else:
            model = nn.DataParallel(model)

    return model