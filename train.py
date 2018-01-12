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

from torchtext.vocab import Vocab
# from torchtext.vocab import GloVe
from torchtext.data import Field, Pipeline, RawField, Dataset, Example, BucketIterator
from torchtext.data import get_tokenizer
import os, time, sys, datetime, argparse, pickle

from model import EncoderRNN, DecoderRNN
import config

EOS = "<eos>"
SOS = "<sos>"
PAD = "<pad>"
np.random.seed(666)


def stoi(s, field):
    sent = [field.vocab.stoi[w] for w in s]
    return sent

def itos(s, field):
    sent = " ".join([field.vocab.itos[w] for w in s])
    return sent.strip()

def since(t):
    return '[' + str(datetime.timedelta(seconds=time.time() - t)) + '] '


def load_data(c):
    """
    Load datasets, return a dictionary of datasets and fields
    """

    # TODO: add field for context

    spacy_src = spacy.load(c['src_lang'])
    spacy_trg = spacy.load(c['trg_lang'])

    def tokenize_src(text):
        return [tok.text for tok in spacy_src.tokenizer(text)]

    def tokenize_trg(text):
        return [tok.text for tok in spacy_trg.tokenizer(text)]

    src_field = Field(tokenize=tokenize_src, include_lengths=True, eos_token=EOS, lower=True)
    trg_field= Field(tokenize=tokenize_trg, include_lengths=True, eos_token=EOS, lower=True, init_token=SOS)

    datasets = {}
    # load processed data
    for split in c['splits']:
        if os.path.isfile(c['root'] + split + '.pkl'):
            print('Loading {0}'.format(c['root'] + split + '.pkl'))
            examples = pickle.load(open(c['root'] + split + '.pkl', 'rb'))
            datasets[split] = Dataset(examples = examples, fields={'src':src_field,'trg': trg_field})
        else:
            src_path = c['root'] + split + '.src'
            trg_path = c['root'] + split + '.trg'
            examples = c['load'](src_path, trg_path, src_field, trg_field)
            datasets[split] = Dataset(examples = examples, fields={'src':src_field,'trg': trg_field})
            print('Saving to {0}'.format(c['root'] + split + '.pkl'))
            pickle.dump(examples, open(c['root'] + split + '.pkl', 'wb'))

    return datasets, src_field, trg_field

def evaluate(encoder, decoder, var, trg_field, max_len=30, beam_size=-1):
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
    decoder_inputs, decoder_lenghts = trg_field.numericalize(([[SOS]], [1]), device=-1)

    if beam_size > 0:
        for i in range(max_len):
            for h in H:
                hyp, s = h
                decoder_inputs, decoder_lenghts = trg_field.numericalize(([hyp], [len(hyp)]), device=-1)
                decoder_unpacked, decoder_hidden = decoder(decoder_inputs, decoder_hidden, encoder_unpacked, encoder_lengths)
                topv, topi = decoder_unpacked.data[-1].topk(beam_size)
                topv = logsm(topv)
                for j in range(beam_size):
                    nj = int(topi.numpy()[0][j])
                    hyp_new = hyp + [trg_field.vocab.itos[nj]]
                    s_new = s + topv.data.numpy().tolist()[-1][j]
                    if trg_field.vocab.itos[nj] == EOS:
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
            if trg_field.vocab.itos[ni] == EOS:
                outputs.append(EOS)
                break
            else:
                outputs.append(trg_field.vocab.itos[ni])
            decoder_inputs = Variable(torch.LongTensor([[ni]]))
        outputs = " ".join(outputs)
    return outputs.strip()


def random_eval(encoder, decoder, batch, n, src_field, trg_field, beam_size=-1):
    print("Random sampling...")
    enc_inputs, enc_lengths = batch.src
    dec_inputs, dec_lengths = batch.trg
    N = enc_inputs.size()[1]
    idx = np.random.choice(N, n)
    for i in idx:
        print('\t> ' + itos(enc_inputs[:,i].data.numpy(), src_field))
        print('\t= ' + itos(dec_inputs[:,i].data.numpy(), trg_field))
        eval_input = (enc_inputs[:,i].unsqueeze(1), torch.LongTensor([enc_lengths[i]]))
        sent = evaluate(encoder, decoder, eval_input, trg_field=trg_field, beam_size=beam_size)
        print('\t< ' + sent)
        print()


def main(args):
    start = time.time()
    print(since(start) + "Loading data with configuration '{0}'...".format(args.config))
    c = getattr(config, args.config)()
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

    PAD_IDX = trg_field.vocab.stoi[PAD] # default=1
    if args.from_scratch or not os.path.isfile(c['model_path'] + c['prefix'] + 'encoder.pkl') \
            or not os.path.isfile(c['model_path'] + c['prefix'] + 'decoder.pkl'):
        encoder = EncoderRNN(vocab_size=len(src_field.vocab), embed_size=c['encoder_embed_size'],\
                hidden_size=c['encoder_hidden_size'], padding_idx=PAD_IDX, n_layers=c['num_layers'])
        decoder = DecoderRNN(vocab_size=len(trg_field.vocab), embed_size=c['decoder_embed_size'],\
                hidden_size=c['decoder_hidden_size'], encoder_hidden=c['encoder_hidden_size'],\
                padding_idx=PAD_IDX, n_layers=c['num_layers'])
    else:
        print(since(start) + "Loading models...")
        encoder = torch.load(c['model_path'] + c['prefix'] + 'encoder.pkl')
        decoder = torch.load(c['model_path'] + c['prefix'] + 'decoder.pkl')

    CEL = nn.CrossEntropyLoss(size_average=True, ignore_index=PAD_IDX)
    params = list(encoder.parameters()) +  list(decoder.parameters())
    optimizer = optim.Adam(params, lr=c['learning_rate'])
    print_loss = 0

    train_iter = BucketIterator(
        dataset=train, batch_size=c['batch_size'],
        sort_key=lambda x: -len(x.src), device=-1)

    test_iter = BucketIterator(
        dataset=test, batch_size=c['batch_size'],
        sort_key=lambda x: -len(x.src), device=-1)

    print(since(start) + "Start training... {0} iterations...".format(n_iters))

    for i in range(1, n_iters+1):
        batch = next(iter(train_iter))
        encoder_inputs, encoder_lengths = batch.src
        decoder_inputs, decoder_lengths = batch.trg
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
            print(since(start) + "Saving models...")
            torch.save(encoder, c['model_path'] + c['prefix'] + 'encoder.pkl')
            torch.save(decoder, c['model_path'] + c['prefix'] + 'decoder.pkl')


        if i % c['log_step'] == 0:
            # TODO: performence on test dataset
            print(since(start) + 'iteration {0}/{1}'.format(i, n_iters))
            print("\tLoss: ", print_loss.data.numpy().tolist()[0] / c['log_step'])
            print_loss = 0
            random_eval(encoder, decoder, batch, n=1, src_field=src_field, trg_field=trg_field, beam_size=c['beam_size'])





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None ,
                        help='model configurations, defined in config.py')
    parser.add_argument('--from_scratch', type=bool, default=False)
    parser.add_argument('--disable_cuda', type=bool, default=False)
    args = parser.parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    main(args)
