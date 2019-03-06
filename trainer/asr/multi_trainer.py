import time
from tqdm import tqdm
from utils import constant
from utils.functions import save_model
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer
import numpy as np

import torch
import sys


class Trainer():
    """
    Dataset Trainer class for Joint-Training
    """

    def __init__(self):
        print("Trainer is initialized")

    def train(self, model, train_loader, train_sampler, valid_loaders, opt, loss_type, start_epoch, num_epochs, label2id, id2label, last_metrics=None, logger=None):
        """
        Training
        args:
            model: Model object
            train_loader: DataLoader object of the training set
            valid_loaders: list of DataLoader object of the validation set
            opt: Optimizer object
            start_epoch: start epoch (> 0 if you resume the process)
            num_epochs: last epoch
            last_metrics: (if resume)
        """
        if logger is not None:
            sys.out = logger

        start_time = time.time()
        best_valid_loss = 1000000000 if last_metrics is None else last_metrics['valid_loss']
        smoothing = constant.args.label_smoothing

        history = []

        for epoch in range(start_epoch, num_epochs):
            sys.out.flush()
            total_loss, total_cer, total_wer, total_char, total_word = 0, 0, 0, 0, 0
            start_iter = 0

            print("TRAIN")
            model.train()
            pbar = tqdm(iter(train_loader), leave=True,
                        total=len(train_loader))
            for i, (data) in enumerate(pbar, start=start_iter):
                src, tgt, src_percentages, src_lengths, tgt_lengths = data

                if constant.USE_CUDA:
                    src = src.cuda()
                    tgt = tgt.cuda()

                opt.optimizer.zero_grad()

                pred, gold, hyp_seq, gold_seq = model(
                    src, input_lengths=src_lengths, padded_target=tgt, verbose=constant.args.verbose)

                strs_gold = ["".join([id2label[int(x)]
                                      for x in gold]) for gold in gold_seq]
                strs_hyps = ["".join([id2label[int(x)]
                                      for x in hyp]) for hyp in hyp_seq]

                loss, num_correct = calculate_metrics(
                    pred, gold, smoothing=smoothing, loss_type=loss_type, input_lengths=src_lengths, target_lengths=tgt_lengths)

                if constant.args.verbose:
                    print("GOLD", strs_gold)
                    print("HYP", strs_hyps)

                for j in range(len(strs_hyps)):
                    cer = calculate_cer(strs_hyps[j], strs_gold[j])
                    wer = calculate_wer(strs_hyps[j], strs_gold[j])
                    total_cer += cer
                    total_wer += wer
                    total_char += len(strs_gold[j])
                    total_word += len(strs_gold[j].split(" "))

                loss.backward()
                opt.optimizer.step()

                total_loss += loss.detach().item()
                non_pad_mask = gold.ne(constant.PAD_TOKEN)
                num_word = non_pad_mask.sum().item()

                pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} CER:{:.2f}% WER:{:.2f}%".format(
                    (epoch+1), total_loss/(i+1), total_cer*100/total_char, total_wer*100/total_word))
            print("(Epoch {}) TRAIN LOSS:{:.4f} CER:{:.2f}% WER:{:.2f}%".format(
                (epoch+1), total_loss/(len(train_loader)), total_cer*100/total_char, total_wer*100/total_word))

            print("VALID")
            all_valid_loss = []
            for valid_task_id in range(len(valid_loaders)):
                model.eval()
                sys.out.flush()

                valid_loader = valid_loaders[valid_task_id]

                total_valid_loss, total_valid_cer, total_valid_wer, total_valid_char, total_valid_word = 0, 0, 0, 0, 0
                valid_pbar = tqdm(iter(valid_loader), leave=True,
                                  total=len(valid_loader))
                for i, (data) in enumerate(valid_pbar):
                    src, tgt, src_percentages, src_lengths, tgt_lengths = data

                    if constant.USE_CUDA:
                        src = src.cuda()
                        tgt = tgt.cuda()

                    pred, gold, hyp_seq, gold_seq = model(
                        src, input_lengths=src_lengths, padded_target=tgt, verbose=constant.args.verbose)
                    loss, num_correct = calculate_metrics(
                        pred, gold, smoothing=smoothing, loss_type=loss_type, input_lengths=src_lengths, target_lengths=tgt_lengths)

                    strs_gold = ["".join([id2label[int(x)]
                                          for x in gold]) for gold in gold_seq]
                    strs_hyps = ["".join([id2label[int(x)]
                                          for x in hyp]) for hyp in hyp_seq]

                    for j in range(len(strs_hyps)):
                        cer = calculate_cer(strs_hyps[j], strs_gold[j])
                        wer = calculate_wer(strs_hyps[j], strs_gold[j])
                        total_valid_cer += cer
                        total_valid_wer += wer
                        total_valid_char += len(strs_gold[j])
                        total_valid_word += len(strs_gold[j].split(" "))

                    total_valid_loss += loss.detach().item()
                    valid_pbar.set_description("(Epoch {}) TASK:{} VALID LOSS:{:.4f} CER:{:.2f}% WER:{:.2f}%".format(
                        (epoch+1), valid_task_id, total_valid_loss/(i+1), total_valid_cer*100/total_valid_char, total_valid_wer*100/total_valid_word))
                all_valid_loss.append(total_valid_loss / len(valid_pbar))
                print("(Epoch {}) TASK:{} VALID LOSS:{:.4f} CER:{:.2f}% WER:{:.2f}%".format(
                    (epoch+1), valid_task_id, total_valid_loss/(len(valid_loader)), total_valid_cer*100/total_valid_char, total_valid_wer*100/total_valid_word))

            metrics = {}
            metrics["train_loss"] = total_loss / len(train_loader)
            metrics["valid_loss"] = np.mean(np.array(all_valid_loss))
            metrics["valid_losses"] = all_valid_loss
            metrics["train_cer"] = total_cer
            metrics["train_wer"] = total_wer
            metrics["valid_cer"] = total_valid_cer
            metrics["valid_wer"] = total_valid_wer
            metrics["history"] = history
            history.append(metrics)

            if epoch % constant.args.save_every == 0:
                save_model(model, (epoch+1), opt, metrics,
                           label2id, id2label, best_model=False)

            # save the best model
            if best_valid_loss > total_valid_loss/len(valid_loader):
                best_valid_loss = total_valid_loss/len(valid_loader)
                save_model(model, (epoch+1), opt, metrics,
                           label2id, id2label, best_model=True)

            if constant.args.shuffle:
                print("SHUFFLE")
                train_sampler.shuffle(epoch)
