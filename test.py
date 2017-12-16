#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Yifan WANG <yifanwang1993@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np
import spacy
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import Counter, OrderedDict

from torchtext import data, datasets
from torchtext.vocab import Vocab
# from torchtext.vocab import GloVe
from torchtext.data import Field, Pipeline, RawField, Dataset
from torchtext.data import get_tokenizer
import os, time, sys, datetime, argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from model import EncoderRNN, DecoderRNN

EOS = "<eos>"
SOS = "<sos>"
PAD = "<pad>"
np.random.seed(666)

def stoi(s, en, fr, lang='fr'):
    if lang == 'fr':
        sent = [fr.vocab.stoi[w] for w in s]
    else:
        sent = [en.vocab.stoi[w] for w in s]
    return sent

def itos(s, en, fr, lang='en'):
    if lang == 'en':
        sent = " ".join([en.vocab.itos[w] for w in s])
    else:
        sent = " ".join([fr.vocab.itos[w] for w in s])
    return sent.strip()

def since(t):
    return '[' + str(datetime.timedelta(seconds=time.time() - t)) + '] '


def load_data():
    # Read data
    with open('data/eng-fra.txt', 'r') as f:
        en = open('data/trans.en', 'w')
        fr = open('data/trans.fr', 'w')
        for l in f.readlines():
            fr.write(l.split('\t')[1].strip() + '\n')
            en.write(l.split('\t')[0].strip() + '\n')
        en.close()
        fr.close()

    spacy_fr = spacy.load('fr')
    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_fr(text):
        return [tok.text for tok in spacy_fr.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    # french -> english
    en = data.Field(tokenize=tokenize_en, include_lengths=True, eos_token=EOS, init_token=SOS, lower=True)
    fr = data.Field(tokenize=tokenize_fr, include_lengths=True, eos_token=EOS, lower=True)
    trans = datasets.TranslationDataset(
        path='data/trans', exts=('.fr', '.en'),
        fields=(fr, en))
    en.build_vocab(trans)
    fr.build_vocab(trans)
    return trans, en, fr

def evaluate(encoder, decoder, var, dec_field, max_len=30, beam_size=-1):
    """
    var: tuple of tensors
    """
    logsm = nn.LogSoftmax()
    # Beam search
    # TODO: check the beam search
    H = [([SOS], 0.)]
    H_temp = []
    H_final = []

    outputs = []
    encoder_inputs, encoder_lengths = var
    encoder_packed, encoder_hidden = encoder(encoder_inputs, encoder_lengths)
    encoder_unpacked = pad_packed_sequence(encoder_packed)[0]
    decoder_hidden = encoder_hidden
    decoder_inputs, decoder_lenghts = dec_field.numericalize(([[SOS]], [1]), device=-1)

    if beam_size > 0:
        for i in range(max_len):
            for h in H:
                hyp, s = h
                decoder_inputs, decoder_lenghts = dec_field.numericalize(([hyp], [len(hyp)]), device=-1)
                decoder_unpacked, decoder_hidden = decoder(decoder_inputs, decoder_hidden, encoder_unpacked, encoder_lengths)
                topv, topi = decoder_unpacked.data[-1].topk(beam_size)
                topv = logsm(topv)
                for j in range(beam_size):
                    nj = int(topi.numpy()[0][j])
                    hyp_new = hyp + [dec_field.vocab.itos[nj]]
                    s_new = s + topv.data.numpy().tolist()[-1][j]
                    if dec_field.vocab.itos[nj] == EOS:
                        H_final.append((hyp_new, s_new))
                    else:
                        H_temp.append((hyp_new, s_new))
                H_temp = sorted(H_temp, key=lambda x:x[1], reverse=True)
                H = H_temp[:beam_size]
                H_temp = []

        H_final = sorted(H_final, key=lambda x:x[1], reverse=True)
        outputs = [" ".join(H_final[i][0]) for i in range(beam_size)]

    else:
        for i in range(max_len):
            # Eval mode, dropout is not used
            decoder_unpacked, decoder_hidden = decoder.eval()(decoder_inputs, decoder_hidden, encoder_unpacked, encoder_lengths)
            topv, topi = decoder_unpacked.data.topk(1)
            ni = int(topi.numpy()[0][0][0])
            if dec_field.vocab.itos[ni] == EOS:
                outputs.append(EOS)
                break
            else:
                outputs.append(dec_field.vocab.itos[ni])
            decoder_inputs = Variable(torch.LongTensor([[ni]]))
        outputs = " ".join(outputs)
    return outputs.strip()


def main(args):
    start = time.time()
    print(since(start) + "Loading data...")
    trans, en, fr = load_data()
    N = len(trans.examples)
    PAD_IDX = fr.vocab.stoi[PAD] # default=1

    print(since(start) + "Loading models...")
    encoder = torch.load(args.model_path + 'encoder.pkl')
    decoder = torch.load(args.model_path + 'decoder.pkl')
    spacy_fr = spacy.load('fr')
    sent = input(">> ")
    while sent.strip() != ':q':
        tokenized = [tok.text for tok in spacy_fr.tokenizer(sent.strip().lower())] + [EOS]
        inputs, lenghts = fr.numericalize(([tokenized], [len(tokenized)]), device=-1)
        outputs = evaluate(encoder, decoder,(inputs, lenghts), dec_field=en, beam_size=args.beam_size)
        print("<< " + outputs + '\n')
        sent = input(">> ")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--beam_size', type=int , default=-1,
                        help='beam size')

    args = parser.parse_args()
    print(args)
    main(args)
