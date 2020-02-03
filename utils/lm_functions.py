import torch
import os
import math
import torch.nn as nn

from models.lm.transformer_lm import TransformerLM
from utils.optimizer import NoamOpt
from utils import constant


# def save_model(model, epoch, opt, metrics, label2id, id2label, best_model=False):
#     """
#     Saving model, TODO adding history
#     """
#     if best_model:
#         save_path = "{}/{}/best_model.th".format(
#             constant.args.save_folder, constant.args.name)
#     else:
#         save_path = "{}/{}/epoch_{}.th".format(constant.args.save_folder,
#                                                constant.args.name, epoch)

#     if not os.path.exists(constant.args.save_folder + "/" + constant.args.name):
#         os.makedirs(constant.args.save_folder + "/" + constant.args.name)

#     print("SAVE MODEL to", save_path)
#     args = {
#         'label2id': label2id,
#         'id2label': id2label,
#         'args': constant.args,
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': opt.optimizer.state_dict(),
#         'optimizer_params': {
#             '_step': opt._step,
#             '_rate': opt._rate,
#             'warmup': opt.warmup,
#             'factor': opt.factor,
#             'model_size': opt.model_size
#         },
#         'metrics': metrics
#     }
#     torch.save(args, save_path)


# def load_model(load_path):
#     """
#     Loading model
#     args:
#         load_path: string
#     """
#     checkpoint = torch.load(load_path)

#     epoch = checkpoint['epoch']
#     metrics = checkpoint['metrics']
#     if 'args' in checkpoint:
#         args = checkpoint['args']

#     label2id = checkpoint['label2id']
#     id2label = checkpoint['id2label']

#     model = init_transformer_model(args, label2id, id2label)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     if args.cuda:
#         model = model.cuda()

#     opt = init_optimizer(args, model)
#     if opt is not None:
#         opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         opt._step = checkpoint['optimizer_params']['_step']
#         opt._rate = checkpoint['optimizer_params']['_rate']
#         opt.warmup = checkpoint['optimizer_params']['warmup']
#         opt.factor = checkpoint['optimizer_params']['factor']
#         opt.model_size = checkpoint['optimizer_params']['model_size']

#     return model, opt, epoch, metrics, args, label2id, id2label


# def init_optimizer(args, model):
#     dim_input = args.dim_input
#     warmup = args.warmup
#     lr = args.lr

#     opt = NoamOpt(dim_input, 1, warmup, torch.optim.Adam(
#         model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9))

#     return opt


def init_transformer_model(args, label2id, id2label):
    """
    Initiate a new transformer object
    """

    if args.emb_cnn:
        hidden_size = int(math.floor(
            (args.sample_rate * args.window_size) / 2) + 1)
        hidden_size = int(math.floor(hidden_size - 41) / 2 + 1)
        hidden_size = int(math.floor(hidden_size - 21) / 2 + 1)
        hidden_size *= 32
        args.dim_input = hidden_size

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

    model = TransformerLM(id2label, num_src_vocab=len(label2id), num_trg_vocab=len(label2id), num_layers=num_layers, dim_emb=dim_emb, dim_model=dim_model, dim_inner=dim_inner, num_heads=num_heads, dim_key=dim_key, dim_value=dim_value, dropout=dropout)

    return model
