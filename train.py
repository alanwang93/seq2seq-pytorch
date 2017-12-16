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

def split_data(root, file_name, train=0.6, test=0.2):
    if not os.path.isfile(root + 'train.en'):
        valid = 1 - train - test
        with open(path + file_name, 'r') as f:
            lines = f.readlines()
            n = len(lines)
            p = np.random.permutation(n)
            train, test, valid = np.split(lines, int(n*train), int(n*train+n*test))
            for samples, mode in [(train, 'train'), (test, 'test'), (valid, 'valid')]:
                en = open(root + mode + 'en', 'w')
                fr = open(root + mode + 'fr', 'w')
                for l in samples:
                    fr.write(l.split('\t')[1].strip() + '\n')
                    en.write(l.split('\t')[0].strip() + '\n')
                    en.close()
                    fr.close()
            print("Train: {0}\nTest: {1}\nValidation: {2}".format(len(train), len(test), len(valid)))



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
                outputs.append('<EOS>')
                break
            else:
                outputs.append(dec_field.vocab.itos[ni])
            decoder_inputs = Variable(torch.LongTensor([[ni]]))
        outputs = " ".join(outputs)
    return outputs.strip()


def random_eval(encoder, decoder, batch, n, en, fr, beam_size=-1):
    print("Random sampling...")
    enc_inputs, enc_lengths = batch.src
    dec_inputs, dec_lengths = batch.trg
    N = enc_inputs.size()[1]
    idx = np.random.choice(N, n)
    for i in idx:
        print('\t> ' + itos(enc_inputs[:,i].data.numpy(), en, fr, lang='fr'))
        print('\t= ' + itos(dec_inputs[:,i].data.numpy(), en, fr))
        eval_input = (enc_inputs[:,i].unsqueeze(1), torch.LongTensor([enc_lengths[i]]))
        sent = evaluate(encoder, decoder, eval_input, dec_field=en, beam_size=beam_size)
        print('\t< ' + sent)
        print()


def main(args):
    # TODO: split dataset into 3 parts
    start = time.time()
    print(since(start) + "Loading data...")
    trans, en, fr = load_data()
    N = len(trans.examples)
    batch_per_epoch = N // args.batch_size
    n_iters = batch_per_epoch * args.num_epochs
    print(since(start) + "{0} training samples, batch size={1}, {2} batches per epoch.".format(N, args.batch_size, batch_per_epoch))

    PAD_IDX = fr.vocab.stoi[PAD] # default=1
    if args.from_scratch or not os.path.isfile(args.model_path + 'encoder.pkl') \
            or not os.path.isfile(args.model_path + 'decoder.pkl'):
        encoder = EncoderRNN(vocab_size=len(fr.vocab), embed_size=args.encoder_embed_size,\
                hidden_size=args.encoder_hidden_size, padding_idx=PAD_IDX)
        decoder = DecoderRNN(vocab_size=len(en.vocab), embed_size=args.decoder_embed_size,\
                hidden_size=args.decoder_hidden_size, encoder_hidden=args.encoder_hidden_size,\
                padding_idx=PAD_IDX)
    else:
        print(since(start) + "Loading models...")
        encoder = torch.load(args.model_path + 'encoder.pkl')
        decoder = torch.load(args.model_path + 'decoder.pkl')

    CEL = nn.CrossEntropyLoss(size_average=True, ignore_index=PAD_IDX)
    params = list(encoder.parameters()) +  list(decoder.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate)
    print_loss = 0

    train_iter = data.BucketIterator(
        dataset=trans, batch_size=args.batch_size,
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
        tgt_len, batch_size, d = decoder_unpacked.size()
        # remove first symbol <SOS>
        loss = CEL(decoder_unpacked.view(tgt_len*batch_size, d), decoder_inputs[1:,:].view(-1))
        print_loss += loss

        if i % args.save_step == 0:
            print(since(start) + "Saving models...")
            torch.save(encoder, args.model_path + 'encoder.pkl')
            torch.save(decoder, args.model_path + 'decoder.pkl')


        if i % args.log_step == 0:
            print(since(start) + 'iteration {0}'.format(i))
            print("\tLoss: ", print_loss.data.numpy().tolist()[0] / args.log_step)
            print_loss = 0
            random_eval(encoder, decoder, batch, n=1, en=en, fr=fr, beam_size=-1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--log_step', type=int , default=50,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=500,
                        help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--encoder_embed_size', type=int , default=256 ,
                        help='dimension of encoder word embedding vectors')
    parser.add_argument('--decoder_embed_size', type=int , default=256 ,
                        help='dimension of decoder word embedding vectors')
    parser.add_argument('--encoder_hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--decoder_hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    # parser.add_argument('--num_layers', type=int , default=1 ,
    #                     help='number of layers in rnn')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--from_scratch', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    args = parser.parse_args()
    print(args)
    main(args)
