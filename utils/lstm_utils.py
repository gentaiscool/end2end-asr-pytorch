import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import constant
from data.helper import get_word_segments_per_language, is_contain_chinese_word
from torch.autograd import Variable
import math

def calculate_lm_score(seq, lm, id2label):
    """
    seq: (1, seq_len)
    id2label: map
    """
    # print("hello")
    seq_str = "".join(id2label[char.item()] for char in seq[0]).replace(
        constant.PAD_CHAR, "").replace(constant.SOS_CHAR, "").replace(constant.EOS_CHAR, "")
    seq_str = seq_str.replace("  ", " ")

    seq_arr = get_word_segments_per_language(seq_str)
    seq_str = ""
    for i in range(len(seq_arr)):
        if is_contain_chinese_word(seq_arr[i]):
            for char in seq_arr[i]:
                if seq_str != "":
                    seq_str += " "
                seq_str += char
        else:
            if seq_str != "":
                seq_str += " "
            seq_str += seq_arr[i]

    # print("seq_str:", seq_str)
    seq_str = seq_str.replace("  ", " ").replace("  ", " ")
    # print("seq str:", seq_str)

    if seq_str == "":
        return -999, 0, 0

    score, oov_token = lm.evaluate(seq_str)    
    
    # a, b = lm.evaluate("除非 的 不会 improve 什么 东西 的 这些 esperience")
    # a2, b2 = lm.evaluate("除非 的 不会 improve 什么 东西 的 这些 experience")
    # print(a, a2)
    return -1 * score / len(seq_str.split()) + 1, len(seq_str.split()) + 1, oov_token


class LM(object):
    def __init__(self, model_path):
        self.model_path = model_path
        print("load model path:", self.model_path)

        checkpoint = torch.load(model_path)
        self.word2idx = checkpoint["word2idx"]
        self.idx2word = checkpoint["idx2word"]
        ntokens = checkpoint["ntoken"]
        ninp = checkpoint["ninp"]
        nhid = checkpoint["nhid"]
        nlayers = checkpoint["nlayers"]
        dropout = checkpoint["dropout"]
        tie_weights = checkpoint["tie_weights"]

        self.model = RNNModel("LSTM", ntoken=ntokens, ninp=ninp, nhid=nhid,
                              nlayers=nlayers, dropout=dropout, tie_weights=tie_weights)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if constant.args.cuda:
            self.model = self.model.cuda()

        self.criterion = nn.CrossEntropyLoss()

    def batchify(self, data, bsz, cuda):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        if cuda:
            data = data.cuda()
        return data

    def seq_to_tensor(self, seq):
        words = seq.split() + ['<eos>']

        ids = torch.LongTensor(len(words))
        token = 0
        oov_token = 0
        for word in words:
            if word in self.word2idx:
                ids[token] = self.word2idx[word]
            else:
                ids[token] = self.word2idx['<oov>']
                oov_token += 1
            # print(">", word, ids[token])
            token += 1
        # print("ids", ids)
        return ids, oov_token

    def get_batch(self, source, i, bptt, seq_len=None, evaluation=False):
        seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

    def evaluate(self, seq):
        """
        batch_size = 1
        """
        tensor, oov_token = self.seq_to_tensor(seq)
        data_source = self.batchify(tensor
            , 1, constant.args.cuda)
        self.model.eval()

        total_loss = 0
        ntokens = len(self.word2idx)
        hidden = self.model.init_hidden(1)
        data, targets = self.get_batch(
            data_source, 0, data_source.size(0), evaluation=True)
        output, hidden = self.model(data, hidden)

        # calculate probability
        # print(output.size()) # seq_len, vocab

        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * self.criterion(output_flat, targets).data
        hidden = self.repackage_hidden(hidden)
        return total_loss, oov_token

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors,
        to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh',
                                'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers,
                              nonlinearity=nonlinearity, dropout=dropout)

        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))

        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)

        decoded = self.decoder(output.view(
            output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
