from awd_lstm_lm.splitcross import SplitCrossEntropyLoss

import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import constant

def calculate_lm_score(seq, lm, id2label):
    """
    seq: (1, seq_len)
    id2label: map
    """
    # print(seq)
    # for char in seq[0]:
    #     print(char.item())
    seq_str = "".join(id2label[char.item()] for char in seq[0]).replace(constant.PAD_CHAR,"").replace(constant.SOS_CHAR,"").replace(constant.EOS_CHAR,"")
    seq_str = seq_str.replace("  ", " ")
    if seq_str == "":
        return -999
        
    score = lm.evaluate(seq_str)
    # print(seq_str)
    return -1 * (score / len(seq_str.split())), len(seq_str.split())

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
        dropouth = checkpoint["dropouth"]
        dropouti = checkpoint["dropouti"]
        dropoute = checkpoint["dropoute"]
        tie_weights = checkpoint["tie_weights"]

        self.model = RNNModel("LSTM", ntoken=ntokens, ninp=ninp, nhid=nhid, nlayers=nlayers, dropout=dropout, dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=1, tie_weights=tie_weights)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if constant.args.cuda:
            self.model = self.model.cuda()

        splits = []
        if ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        print('Using', splits)
        self.criterion = SplitCrossEntropyLoss(ninp, splits=splits, verbose=False)

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
        for word in words:
            if word in self.word2idx:
                ids[token] = self.word2idx[word]
            else:
                ids[token] = self.word2idx['<oov>']
            token += 1
        return ids

    def get_batch(self, source, i, bptt, seq_len=None, evaluation=False):
        seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

    def evaluate(self, seq):
        """
        batch_size = 1
        """
        data_source = self.batchify(self.seq_to_tensor(seq), 1, constant.args.cuda)
        self.model.eval()

        total_loss = 0
        ntokens = len(self.word2idx)
        hidden = self.model.init_hidden(1)
        data, targets = self.get_batch(
            data_source, 0, data_source.size(0), evaluation=True)
        output, hidden = self.model(data, hidden)

        # calculate probability
        # print(output.size()) # seq_len, vocab

        total_loss = len(data) * self.criterion(self.model.decoder.weight, self.model.decoder.bias, output, targets).data
        hidden = self.repackage_hidden(hidden)
        return total_loss

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors,
        to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

from awd_lstm_lm.embed_regularize import embedded_dropout
from awd_lstm_lm.locked_dropout import LockedDropout
from awd_lstm_lm.weight_drop import WeightDrop

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
