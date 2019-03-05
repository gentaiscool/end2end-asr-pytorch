import torch
import torch.nn.functional as F
import Levenshtein as Lev

from utils import constant

from data.helper import get_word_segments_per_language, is_contain_chinese_word

def calculate_cer_en_zh(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.

    Arguments:
        s1 (string): space-separated sentence (hyp)
        s2 (string): space-separated sentence (gold)
    """
    s1_segments = get_word_segments_per_language(s1)
    s2_segments = get_word_segments_per_language(s2)

    en_s1_seq, en_s2_seq = "", ""
    zh_s1_seq, zh_s2_seq = "", ""

    for segment in s1_segments:
        if is_contain_chinese_word(segment):
            if zh_s1_seq != "":
                zh_s1_seq += " "
            zh_s1_seq += segment
        else:
            if en_s1_seq != "":
                en_s1_seq += " "
            en_s1_seq += segment
    
    for segment in s2_segments:
        if is_contain_chinese_word(segment):
            if zh_s2_seq != "":
                zh_s2_seq += " "
            zh_s2_seq += segment
        else:
            if en_s2_seq != "":
                en_s2_seq += " "
            en_s2_seq += segment

    # print(">", en_s1_seq, "||", en_s2_seq, len(en_s2_seq), "||", calculate_cer(en_s1_seq, en_s2_seq) / max(1, len(en_s2_seq.replace(' ', ''))))
    # print(">>", zh_s1_seq, "||", zh_s2_seq, len(zh_s2_seq), "||", calculate_cer(zh_s1_seq, zh_s2_seq) /  max(1, len(zh_s2_seq.replace(' ', ''))))

    return calculate_cer(en_s1_seq, en_s2_seq), calculate_cer(zh_s1_seq, zh_s2_seq), len(en_s2_seq), len(zh_s2_seq)

def calculate_cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.

    Arguments:
        s1 (string): space-separated sentence (hyp)
        s2 (string): space-separated sentence (gold)
    """
    return Lev.distance(s1, s2)

def calculate_wer(s1, s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """

    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]

    return Lev.distance(''.join(w1), ''.join(w2))

def calculate_metrics(pred, gold, input_lengths=None, target_lengths=None, smoothing=0.0, loss_type="ce"):
    """
    Calculate metrics
    args:
        pred: B x T x C
        gold: B x T
        input_lengths: B (for CTC)
        target_lengths: B (for CTC)
    """
    loss = calculate_loss(pred, gold, input_lengths, target_lengths, smoothing, loss_type)
    if loss_type == "ce":
        pred = pred.view(-1, pred.size(2)) # (B*T) x C
        gold = gold.contiguous().view(-1) # (B*T)
        pred = pred.max(1)[1]
        non_pad_mask = gold.ne(constant.PAD_TOKEN)
        num_correct = pred.eq(gold)
        num_correct = num_correct.masked_select(non_pad_mask).sum().item()
        return loss, num_correct
    elif loss_type == "ctc":
        return loss, None
    else:
        print("loss is not defined")
        return None, None

def calculate_loss(pred, gold, input_lengths=None, target_lengths=None, smoothing=0.0, loss_type="ce"):
    """
    Calculate loss
    args:
        pred: B x T x C
        gold: B x T
        input_lengths: B (for CTC)
        target_lengths: B (for CTC)
        smoothing:
        type: ce|ctc (ctc => pytorch 1.0.0 or later)
        input_lengths: B (only for ctc)
        target_lengths: B (only for ctc)
    """
    if loss_type == "ce":
        pred = pred.view(-1, pred.size(2)) # (B*T) x C
        gold = gold.contiguous().view(-1) # (B*T)
        if smoothing > 0.0:
            eps = smoothing
            num_class = pred.size(1)

            gold_for_scatter = gold.ne(constant.PAD_TOKEN).long() * gold
            one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
            one_hot = one_hot * (1-eps) + (1-one_hot) * eps / num_class
            log_prob = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(constant.PAD_TOKEN)
            num_word = non_pad_mask.sum().item()
            loss = -(one_hot * log_prob).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum() / num_word
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=constant.PAD_TOKEN, reduction="mean")
    elif loss_type == "ctc":
        log_probs = pred.transpose(0, 1) # T x B x C
        # print(gold.size())
        targets = gold
        # targets = gold.contiguous().view(-1) # (B*T)

        """
        log_probs: torch.Size([209, 8, 3793])
        targets: torch.Size([8, 46])
        input_lengths: torch.Size([8])
        target_lengths: torch.Size([8])
        """

        # print("log_probs:", log_probs.size())
        # print("targets:", targets.size())
        # print("input_lengths:", input_lengths.size())
        # print("target_lengths:", target_lengths.size())
        # print(input_lengths)
        # print(target_lengths)

        log_probs = F.log_softmax(log_probs, dim=2)
        loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction="mean")
        # mask = loss.clone() # mask Inf loss
        # # mask[mask != float("Inf")] = 1
        # mask[mask == float("Inf")] = 0

        # loss = mask
        # print(loss)

        # loss_size = len(loss)
        # loss = loss.sum() / loss_size
        # print(loss)
    else:
        print("loss is not defined")

    return loss