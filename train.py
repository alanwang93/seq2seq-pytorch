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
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical
from torchtext.vocab import Vocab
# from torchtext.vocab import GloVe
from torchtext.data import Field, Pipeline, RawField, Dataset, Example, BucketIterator
from torchtext.data import get_tokenizer
import os, time, sys, datetime, argparse, pickle

from model import EncoderRNN, DecoderRNN
import config
from utils import *

EOS = "<eos>"
SOS = "<sos>"
PAD = "<pad>"
np.random.seed(666)


def main(args):
    start = time.time()
    print(since(start) + "Loading data with configuration '{0}'...".format(args.config))
    c = getattr(config, args.config)()
    c['use_cuda'] = args.use_cuda
    datasets, src_field, trg_field = load_data(c)
    # TODO: validation dataset

    train = datasets['train']
    src_field.build_vocab(train, max_size=c['encoder_vocab'])
    trg_field.build_vocab(train, max_size=c['decoder_vocab'])

    if 'test' in c['splits']:
        test = datasets['test']

    N = len(train.examples)
    batch_per_epoch = N // c['batch_size']
    n_iters = batch_per_epoch * c['num_epochs']
    print(since(start) + "{0} training samples, batch size={1}, {2} batches per epoch.".format(N, c['batch_size'], batch_per_epoch))
    train_iter = BucketIterator(
        dataset=train, batch_size=c['batch_size'],
        sort_key=lambda x: -len(x.src), device=-1)

    test_iter = BucketIterator(
        dataset=test, batch_size=c['batch_size'],
        sort_key=lambda x: -len(x.src), device=-1)

    PAD_IDX = trg_field.vocab.stoi[PAD] # default=1


    if args.from_scratch or not os.path.isfile(c['model_path'] + c['prefix'] + 'encoder.pkl') \
            or not os.path.isfile(c['model_path'] + c['prefix'] + 'decoder.pkl'):
        # Train from scratch
        encoder = EncoderRNN(vocab_size=len(src_field.vocab), embed_size=c['encoder_embed_size'],\
                hidden_size=c['encoder_hidden_size'], padding_idx=PAD_IDX, n_layers=c['num_layers'])
        decoder = DecoderRNN(vocab_size=len(trg_field.vocab), embed_size=c['decoder_embed_size'],\
                hidden_size=c['decoder_hidden_size'], encoder_hidden=c['encoder_hidden_size'],\
                padding_idx=PAD_IDX, n_layers=c['num_layers'])
    else:
        # Load from saved model
        print(since(start) + "Loading models...")
        encoder = torch.load(c['model_path'] + c['prefix'] + 'encoder.pkl')
        decoder = torch.load(c['model_path'] + c['prefix'] + 'decoder.pkl')

    if c['use_cuda']:
        encoder.cuda()
        decoder.cuda()

    CEL = nn.CrossEntropyLoss(size_average=True, ignore_index=PAD_IDX)
    params = list(encoder.parameters()) +  list(decoder.parameters())
    optimizer = optim.Adam(params, lr=c['learning_rate'])
    print_loss = 0


    print(since(start) + "Start training... {0} iterations...".format(n_iters))

    for i in range(1, n_iters+1):
        batch = next(iter(train_iter))
        encoder_inputs, encoder_lengths = batch.src
        decoder_inputs, decoder_lengths = batch.trg
        # GPU
        encoder_inputs = cuda(encoder_inputs, c['use_cuda'])
        # encoder_lengths = cuda(encoder_lengths, c['use_cuda'])
        decoder_inputs = cuda(decoder_inputs, c['use_cuda'])
        # decoder_lengths = cuda(decoder_lengths, c['use_cuda'])

        encoder_packed, encoder_hidden = encoder(encoder_inputs, encoder_lengths)
        encoder_unpacked = pad_packed_sequence(encoder_packed)[0]
        # remove last symbol
        decoder_unpacked, decoder_hidden = decoder(decoder_inputs[:-1,:], encoder_hidden, encoder_unpacked, encoder_lengths)
        trg_len, batch_size, d = decoder_unpacked.size()
        # remove first symbol <SOS>
        loss = CEL(decoder_unpacked.view(trg_len*batch_size, d), decoder_inputs[1:,:].view(-1))
        print_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % c['save_step'] == 0:
            # TODO: save log
            print(since(start) + "Saving models...")
            torch.save(encoder, c['model_path'] + c['prefix'] + 'encoder.pkl')
            torch.save(decoder, c['model_path'] + c['prefix'] + 'decoder.pkl')

        if i % c['log_step'] == 0:
            # TODO: performence on test dataset
            print(since(start) + 'iteration {0}/{1}'.format(i, n_iters))
            print("\tLoss: ", print_loss.cpu().data.numpy().tolist()[0] / c['log_step'])
            print_loss = 0
            # enc_inputs, enc_lengths = batch.src
            # dec_inputs, dec_lengths = batch.trg
            # eval_input = (enc_inputs[:,1].unsqueeze(1), torch.LongTensor([enc_lengths[1]]))
            # sent = sample(encoder, decoder, eval_input, trg_field=trg_field, greedy=True)
            random_eval(encoder, decoder, batch, n=1, src_field=src_field, trg_field=trg_field, beam_size=c['beam_size'])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None ,
                        help='model configurations, defined in config.py')
    parser.add_argument('--from_scratch', type=bool, default=False)
    parser.add_argument('--disable_cuda', type=bool, default=False)
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Use GPU...")
    else:
        print("Use CPU...")
    main(args)
