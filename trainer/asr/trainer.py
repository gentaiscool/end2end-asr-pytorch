import time
import numpy as np
from tqdm import tqdm
from utils import constant
from utils.functions import save_model
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer
from torch.autograd import Variable
import torch
import logging

import sys

class Trainer():
    """
    Trainer class
    """
    def __init__(self):
        logging.info("Trainer is initialized")

    def train(self, model, train_loader, train_sampler, valid_loader_list, opt, loss_type, start_epoch, num_epochs, label2id, id2label, last_metrics=None):
        """
        Training
        args:
            model: Model object
            train_loader: DataLoader object of the training set
            valid_loader_list: a list of Validation DataLoader objects
            opt: Optimizer object
            start_epoch: start epoch (> 0 if you resume the process)
            num_epochs: last epoch
            last_metrics: (if resume)
        """
        history = []
        start_time = time.time()
        best_valid_loss = 1000000000 if last_metrics is None else last_metrics['valid_loss']
        smoothing = constant.args.label_smoothing

        logging.info("name " +  constant.args.name)

        for epoch in range(start_epoch, num_epochs):
            sys.stdout.flush()
            total_loss, total_cer, total_wer, total_char, total_word = 0, 0, 0, 0, 0

            start_iter = 0

            logging.info("TRAIN")
            model.train()
            pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
            for i, (data) in enumerate(pbar, start=start_iter):
                src, tgt, src_percentages, src_lengths, tgt_lengths = data

                if constant.USE_CUDA:
                    src = src.cuda()
                    tgt = tgt.cuda()

                opt.zero_grad()

                pred, gold, hyp_seq, gold_seq = model(src, src_lengths, tgt, verbose=False)

                try: # handle case for CTC
                    strs_gold, strs_hyps = [], []
                    for ut_gold in gold_seq:
                        str_gold = ""
                        for x in ut_gold:
                            if int(x) == constant.PAD_TOKEN:
                                break
                            str_gold = str_gold + id2label[int(x)]
                        strs_gold.append(str_gold)
                    for ut_hyp in hyp_seq:
                        str_hyp = ""
                        for x in ut_hyp:
                            if int(x) == constant.PAD_TOKEN:
                                break
                            str_hyp = str_hyp + id2label[int(x)]
                        strs_hyps.append(str_hyp)
                except Exception as e:
                    print(e)
                    logging.info("NaN predictions")
                    continue

                seq_length = pred.size(1)
                sizes = Variable(src_percentages.mul_(int(seq_length)).int(), requires_grad=False)

                loss, num_correct = calculate_metrics(
                    pred, gold, input_lengths=sizes, target_lengths=tgt_lengths, smoothing=smoothing, loss_type=loss_type)

                if loss.item() == float('Inf'):
                    logging.info("Found infinity loss, masking")
                    loss = torch.where(loss != loss, torch.zeros_like(loss), loss) # NaN masking
                    continue

                # if constant.args.verbose:
                #     logging.info("GOLD", strs_gold)
                #     logging.info("HYP", strs_hyps)

                for j in range(len(strs_hyps)):
                    strs_hyps[j] = strs_hyps[j].replace(constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
                    strs_gold[j] = strs_gold[j].replace(constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
                    cer = calculate_cer(strs_hyps[j].replace(' ', ''), strs_gold[j].replace(' ', ''))
                    wer = calculate_wer(strs_hyps[j], strs_gold[j])
                    total_cer += cer
                    total_wer += wer
                    total_char += len(strs_gold[j].replace(' ', ''))
                    total_word += len(strs_gold[j].split(" "))

                loss.backward()

                if constant.args.clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), constant.args.max_norm)
                
                opt.step()

                total_loss += loss.item()
                non_pad_mask = gold.ne(constant.PAD_TOKEN)
                num_word = non_pad_mask.sum().item()

                pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} CER:{:.2f}% LR:{:.7f}".format(
                    (epoch+1), total_loss/(i+1), total_cer*100/total_char, opt._rate))
            logging.info("(Epoch {}) TRAIN LOSS:{:.4f} CER:{:.2f}% LR:{:.7f}".format(
                (epoch+1), total_loss/(len(train_loader)), total_cer*100/total_char, opt._rate))

            # evaluate
            print("")
            logging.info("VALID")
            model.eval()

            for ind in range(len(valid_loader_list)):
                valid_loader = valid_loader_list[ind]

                total_valid_loss, total_valid_cer, total_valid_wer, total_valid_char, total_valid_word = 0, 0, 0, 0, 0
                valid_pbar = tqdm(iter(valid_loader), leave=True, total=len(valid_loader))
                for i, (data) in enumerate(valid_pbar):
                    src, tgt, src_percentages, src_lengths, tgt_lengths = data

                    if constant.USE_CUDA:
                        src = src.cuda()
                        tgt = tgt.cuda()

                    pred, gold, hyp_seq, gold_seq = model(src, src_lengths, tgt, verbose=False)

                    seq_length = pred.size(1)
                    sizes = Variable(src_percentages.mul_(int(seq_length)).int(), requires_grad=False)

                    loss, num_correct = calculate_metrics(
                        pred, gold, input_lengths=sizes, target_lengths=tgt_lengths, smoothing=smoothing, loss_type=loss_type)

                    if loss.item() == float('Inf'):
                        logging.info("Found infinity loss, masking")
                        loss = torch.where(loss != loss, torch.zeros_like(loss), loss) # NaN masking
                        continue

                    try: # handle case for CTC
                        strs_gold, strs_hyps = [], []
                        for ut_gold in gold_seq:
                            str_gold = ""
                            for x in ut_gold:
                                if int(x) == constant.PAD_TOKEN:
                                    break
                                str_gold = str_gold + id2label[int(x)]
                            strs_gold.append(str_gold)
                        for ut_hyp in hyp_seq:
                            str_hyp = ""
                            for x in ut_hyp:
                                if int(x) == constant.PAD_TOKEN:
                                    break
                                str_hyp = str_hyp + id2label[int(x)]
                            strs_hyps.append(str_hyp)
                    except Exception as e:
                        print(e)
                        logging.info("NaN predictions")
                        continue

                    for j in range(len(strs_hyps)):
                        strs_hyps[j] = strs_hyps[j].replace(constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
                        strs_gold[j] = strs_gold[j].replace(constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
                        cer = calculate_cer(strs_hyps[j].replace(' ', ''), strs_gold[j].replace(' ', ''))
                        wer = calculate_wer(strs_hyps[j], strs_gold[j])
                        total_valid_cer += cer
                        total_valid_wer += wer
                        total_valid_char += len(strs_gold[j].replace(' ', ''))
                        total_valid_word += len(strs_gold[j].split(" "))

                    total_valid_loss += loss.item()
                    valid_pbar.set_description("VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format(ind,
                        total_valid_loss/(i+1), total_valid_cer*100/total_valid_char))
                logging.info("VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format(ind,
                        total_valid_loss/(len(valid_loader)), total_valid_cer*100/total_valid_char))

            metrics = {}
            metrics["train_loss"] = total_loss / len(train_loader)
            metrics["valid_loss"] = total_valid_loss / (len(valid_loader))
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
                logging.info("SHUFFLE")
                print("SHUFFLE")
                train_sampler.shuffle(epoch)