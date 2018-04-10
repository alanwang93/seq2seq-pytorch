#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Yifan WANG <yifanwang1993@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Sequence to sequence model with global attention.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os, time, sys


class GlobalAttention(nn.Module):
    """
    Global Attention as described in 'Effective Approaches to Attention-based Neural Machine Translation'
    """
    def __init__(self, enc_hidden, dec_hidden):
        super(GlobalAttention, self).__init__()
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden

        # a = h_t^T W h_s
        self.linear_in = nn.Linear(enc_hidden, dec_hidden, bias=False)
        # W [c, h_t]
        self.linear_out = nn.Linear(dec_hidden + enc_hidden, dec_hidden)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def sequence_mask(self, lengths, max_len=None):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return (torch.arange(0, max_len)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))

    def forward(self, inputs, context, context_lengths):
        """
        input (FloatTensor): batch x tgt_len x dim: decoder's rnn's output. (h_t)
        context (FloatTensor): batch x src_len x dim: src hidden states
        context_lengths (LongTensor): the source context lengths.
        """
        # (batch, tgt_len, src_len)
        align = self.score(inputs, context)
        batch, tgt_len, src_len = align.size()


        mask = self.sequence_mask(context_lengths)
        # (batch, 1, src_len)
        mask = mask.unsqueeze(1)  # Make it broadcastable.
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        align.data.masked_fill_(1 - mask, -float('inf')) # fill <pad> with -inf

        align_vectors = self.softmax(align.view(batch*tgt_len, src_len))
        align_vectors = align_vectors.view(batch, tgt_len, src_len)

        # (batch, tgt_len, src_len) * (batch, src_len, enc_hidden) -> (batch, tgt_len, enc_hidden)
        c = torch.bmm(align_vectors, context)

        # \hat{h_t} = tanh(W [c_t, h_t])
        concat_c = torch.cat([c, inputs], 2).view(batch*tgt_len, self.enc_hidden + self.dec_hidden)
        attn_h = self.tanh(self.linear_out(concat_c).view(batch, tgt_len, self.dec_hidden))

        # transpose will make it non-contiguous
        attn_h = attn_h.transpose(0, 1).contiguous()
        align_vectors = align_vectors.transpose(0, 1).contiguous()
        # (tgt_len, batch, dim)
        return attn_h, align_vectors

    def score(self, h_t, h_s):
        """
        h_t (FloatTensor): batch x tgt_len x dim, inputs
        h_s (FloatTensor): batch x src_len x dim, context
        """
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        src_batch, src_len, src_dim = h_s.size()

        h_t = h_t.view(tgt_batch*tgt_len, tgt_dim)
        h_t_ = self.linear_in(h_t)
        h_t = h_t.view(tgt_batch, tgt_len, tgt_dim)
        # (batch, d, s_len)
        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_)


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, padding_idx=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=n_layers)

    def forward(self, inputs, lengths, return_packed=False):
        """
        Inputs:
            inputs: (seq_length, batch_size), non-packed inputs
            lengths: (batch_size)
        """
        # [seq_length, batch_size, embed_length]
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, lengths=lengths.numpy())
        outputs, hiddens = self.rnn(packed)
        if not return_packed:
            return pad_packed_sequence(outputs)[0], hiddens
        return outputs, hiddens


class DecoderRNN(nn.Module):
    """
    """
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, encoder_hidden=None, dropout_p=0.2, padding_idx=1, packed=True):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=n_layers)

        # h_t^T W h_s
        self.linear_out = nn.Linear(hidden_size, vocab_size)
        self.attn = GlobalAttention(encoder_hidden, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs, hidden, context, context_lengths):
        """
        inputs: (tgt_len, batch_size, d)
        hidden: last hidden state from encoder
        context: (src_len, batch_size, hidden_size), outputs of encoder
        """
        # Teacher-forcing, not packed!
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        decoder_unpacked, decoder_hidden = self.rnn(embedded, hidden)
        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            decoder_unpacked.transpose(0, 1).contiguous(),  # (len, batch, d) -> (batch, len, d)
            context.transpose(0, 1).contiguous(),         # (len, batch, d) -> (batch, len, d)
            context_lengths=context_lengths
        )
        # Don't need LogSoftmax with CrossEntropyLoss
        # the outputs are not normalized, and can be negative
        # Note that a mask is needed to compute the loss
        outputs = self.linear_out(attn_outputs)
        return outputs, decoder_hidden
