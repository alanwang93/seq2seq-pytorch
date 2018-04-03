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
    del train
    print("Source vocab: {0}".format(len(src_field.vocab.itos)))
    print("Target vocab: {0}".format(len(trg_field.vocab.itos)))

    test = datasets['test']
    n_test = len(test.examples)

    test_iter = iter(BucketIterator(
        dataset=test, batch_size=1,
        sort_key=lambda x: -len(x.src), device=-1))

    PAD_IDX = trg_field.vocab.stoi[PAD] # default=1

    print(since(start) + "Loading models...")
    encoder = torch.load(c['model_path'] + c['prefix'] + 'encoder.pkl')
    decoder = torch.load(c['model_path'] + c['prefix'] + 'decoder.pkl')

    if c['use_cuda']:
        encoder.cuda()
        decoder.cuda()
    else:
        encoder.cpu()
        decoder.cpu()

    CEL = nn.CrossEntropyLoss(size_average=True, ignore_index=PAD_IDX)
    test_losses = []
    test_rouges = []
    gts = []
    greedys = []
    synchronize(c)
    for i in range(n_test):
        test_batch = next(test_iter)
        test_encoder_inputs, test_encoder_lengths = test_batch.src
        test_decoder_inputs, test_decoder_lengths = test_batch.trg
        test_encoder_inputs = cuda(Variable(test_encoder_inputs.data, volatile=True), c['use_cuda'])
        test_decoder_inputs = cuda(Variable(test_decoder_inputs.data, volatile=True), c['use_cuda'])

        test_encoder_packed, test_encoder_hidden = encoder(test_encoder_inputs, test_encoder_lengths)
        test_encoder_unpacked = pad_packed_sequence(test_encoder_packed)[0]
        # remove last symbol
        test_decoder_unpacked, test_decoder_hidden = decoder(test_decoder_inputs[:-1,:], test_encoder_hidden, test_encoder_unpacked, test_encoder_lengths)
        trg_len, batch_size, d = test_decoder_unpacked.size()
        
        test_loss = CEL(test_decoder_unpacked.view(trg_len*batch_size, d), test_decoder_inputs[1:,:].view(-1))
        
        test_enc_input = (test_encoder_inputs[:,0].unsqueeze(1), torch.LongTensor([test_encoder_lengths[0]]))
        # use self critical training
        test_greedy_out, _ = sample(encoder, decoder, test_enc_input, trg_field,
                max_len=30, greedy=True, config=c)
        test_greedy_sent = tostr(clean(test_greedy_out))

        test_gt_sent = tostr(clean(itos(test_decoder_inputs[:,0].cpu().data.numpy(), trg_field)))

        gts.append(test_gt_sent)
        greedys.append(test_greedy_sent)
        test_rouges.append(score(hyps=test_greedy_sent, refs=test_gt_sent, metric='rouge')['rouge-1']['f'])
        test_losses.append(float(test_loss.cpu().data.numpy().tolist()[0]))
    synchronize(c)
    print("\tTest ROUGE-1_f: ", np.mean(test_rouges))
    print("\tTest Loss: ", np.mean(test_losses))
    
    with open('test.log' ,'w') as f:
        f.write("Test loss: {0}\n".format(np.mean(test_losses)))
        f.write("{0} samples, svg ROUGE-1_f: {1}\n".format(n_test, np.mean(test_rouges)))
        for i in range(n_test):
            f.write(str(test_losses[i]) + '\n')
            f.write(str(test_rouges[i]) + '\n')
            f.write(str(gts[i]) + '\n')
            f.write(str(greedys[i]) + '\n')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None ,
                    help='model configurations, defined in config.py')
    parser.add_argument('--from_scratch', type=bool, default=False)
    parser.add_argument('--disable_cuda', type=bool, default=False)
    parser.add_argument('--self_critical', type=float, default=0.)
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Use GPU...")
    else:
        print("Use CPU...")
    main(args)
