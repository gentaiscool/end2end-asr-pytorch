import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as I
import numpy as np
import math
from utils import constant

"""
General purpose functions
"""

def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    # max_len = max(x.size(0) for x in xs)
    max_len = constant.args.tgt_max_len
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

""" 
Transformer common layers
"""

def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """
    padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # B x T
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    """
    For masking out the padding part of key sequence.
    """
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # B x T_Q x T_K

    return padding_mask

def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """mask position is set to 1"""
    # N x Ti x 1
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class PositionalEncoding(nn.Module):
    """
    Positional Encoding class
    """
    def __init__(self, dim_model, max_length=2000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, dim_model, requires_grad=False)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        exp_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * exp_term) # take the odd (jump by 2)
        pe[:, 1::2] = torch.cos(position * exp_term) # take the even (jump by 2)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        args:
            input: B x T x D
        output:
            tensor: B x T
        """
        return self.pe[:, :input.size(1)]

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feedforward Layer class
    FFN(x) = max(0, xW1 + b1) W2+ b2
    """
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(dim_model, dim_ff)
        self.linear_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        """
        args:
            x: tensor
        output:
            y: tensor
        """
        residual = x
        output = self.dropout(self.linear_2(F.relu(self.linear_1(x))))
        output = self.layer_norm(output + residual)
        return output

class PositionwiseFeedForwardWithConv(nn.Module):
    """
    Position-wise Feedforward Layer Implementation with Convolution class
    """
    def __init__(self, dim_model, dim_hidden, dropout=0.1):
        super(PositionwiseFeedForwardWithConv, self).__init__()
        self.conv_1 = nn.Conv1d(dim_model, dim_hidden, 1)
        self.conv_2 = nn.Conv1d(dim_hidden, dim_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.conv_2(F.relu(self.conv_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_model, dim_key, dim_value, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.dim_model = dim_model
        self.dim_key = dim_key
        self.dim_value = dim_value

        self.query_linear = nn.Linear(dim_model, num_heads * dim_key)
        self.key_linear = nn.Linear(dim_model, num_heads * dim_key)
        self.value_linear = nn.Linear(dim_model, num_heads * dim_value)

        nn.init.normal_(self.query_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_key)))
        nn.init.normal_(self.key_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_key)))
        nn.init.normal_(self.value_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_value)))

        self.attention = ScaledDotProductAttention(temperature=np.power(dim_key, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

        self.output_linear = nn.Linear(num_heads * dim_value, dim_model)
        nn.init.xavier_normal_(self.output_linear.weight)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query: B x T_Q x H, key: B x T_K x H, value: B x T_V x H
        mask: B x T x T (attention mask)
        """
        batch_size, len_query, _ = query.size()
        batch_size, len_key, _ = key.size()
        batch_size, len_value, _ = value.size()

        residual = query

        query = self.query_linear(query).view(batch_size, len_query, self.num_heads, self.dim_key) # B x T_Q x num_heads x H_K
        key = self.key_linear(key).view(batch_size, len_key, self.num_heads, self.dim_key) # B x T_K x num_heads x H_K
        value = self.value_linear(value).view(batch_size, len_value, self.num_heads, self.dim_value) # B x T_V x num_heads x H_V

        query = query.permute(2, 0, 1, 3).contiguous().view(-1, len_query, self.dim_key) # (num_heads * B) x T_Q x H_K
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, len_key, self.dim_key) # (num_heads * B) x T_K x H_K
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, len_value, self.dim_value) # (num_heads * B) x T_V x H_V

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1) # (B * num_head) x T x T
        
        output, attn = self.attention(query, key, value, mask=mask)

        output = output.view(self.num_heads, batch_size, len_query, self.dim_value) # num_heads x B x T_Q x H_V
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_query, -1) # B x T_Q x (num_heads * H_V)

        output = self.dropout(self.output_linear(output)) # B x T_Q x H_O
        output = self.layer_norm(output + residual)

        return output, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """

        """
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

"""
LAS common layers
"""

class DotProductAttention(nn.Module):
    """
    Dot product attention.
    Given a set of vector values, and a vector query, attention is a technique
    to compute a weighted sum of the values, dependent on the query.
    NOTE: Here we use the terminology in Stanford cs224n-2018-lecture11.
    """

    def __init__(self):
        super(DotProductAttention, self).__init__()
        # TODO: move this out of this class?
        # self.linear_out = nn.Linear(dim*2, dim)

    def forward(self, queries, values):
        """
        Args:
            queries: N x To x H
            values : N x Ti x H
        Returns:
            output: N x To x H
            attention_distribution: N x To x Ti
        """
        batch_size = queries.size(0)
        hidden_size = queries.size(2)
        input_lengths = values.size(1)
        # (N, To, H) * (N, H, Ti) -> (N, To, Ti)
        attention_scores = torch.bmm(queries, values.transpose(1, 2))
        attention_distribution = F.softmax(
            attention_scores.view(-1, input_lengths), dim=1).view(batch_size, -1, input_lengths)
        # (N, To, Ti) * (N, Ti, H) -> (N, To, H)
        attention_output = torch.bmm(attention_distribution, values)
        # # concat -> (N, To, 2*H)
        # concated = torch.cat((attention_output, queries), dim=2)
        # # TODO: Move this out of this class?
        # # output -> (N, To, H)
        # output = torch.tanh(self.linear_out(
        #     concated.view(-1, 2*hidden_size))).view(batch_size, -1, hidden_size)

        return attention_output, attention_distribution