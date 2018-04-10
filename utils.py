#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Yifan WANG <yifanwang1993@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Utility functions
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
import os, time, sys, datetime, argparse, pickle, logging
import spacy
from torchtext.vocab import Vocab
# from torchtext.vocab import GloVe
from torchtext.data import Field, Pipeline, RawField, Dataset, Example, BucketIterator
from torchtext.data import get_tokenizer
from rouge import Rouge

# TODO: add these into configuration
EOS = "<eos>"
SOS = "<sos>"
PAD = "<pad>"

def split_data(root, filenames, exts, train_ratio=0.8, test_ratio=0.2):
    """
    Examples: filenames = ['en.txt', 'fr.txt'], exts = ['src', 'trg']
              => train.src, train.trg; test.src, test.trg
    """
    # TODO: check the extension names
    eps = 1e-5
    valid_ratio = 1 - train_ratio - test_ratio
    p = None
    for name, ext in zip(filenames, exts):
        print("Opening {0}".format(name))
        with open(root + name, 'r') as f:
            lines = f.readlines()
            n = len(lines)
            p = np.random.permutation(n) if p is None else p
            train, test, valid = np.split(np.arange(n)[p], [int(n*train_ratio), int(n*train_ratio+n*test_ratio)])

            train = [lines[i] for i in train]
            test = [lines[i] for i in test]
            valid = [lines[i] for i in valid] if valid_ratio > eps else valid
            for samples, mode in [(train, 'train'), (test, 'test'), (valid, 'valid')]:
                if valid_ratio < eps and mode == 'valid':
                    continue
                out = open(root + mode + ext, 'w')
                for l in samples:
                    out.write(l.strip() + '\n')
                out.close()
            print("Train: {0}\nTest: {1}\nValidation: {2}".format(len(train), len(test), len(valid)))


def stoi(s, field):
    sent = [field.vocab.stoi[w] for w in s]
    return sent

def itos(s, field):
    sent = [field.vocab.itos[w] for w in s]
    return sent

def since(t):
    return '[' + str(datetime.timedelta(seconds=time.time() - t)) + '] '

def init_logging(log_name):
    """

    """
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S'   )
    handler = logging.FileHandler(log_name)
    out = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    out.setFormatter(formatter)
    out.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logging.getLogger().addHandler(out)
    logging.getLogger().setLevel(logging.INFO)
    return logging


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


def cuda(var, use_cuda):
    if use_cuda:
        var = var.cuda()
    return var


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
    use_cuda = next(encoder.parameters()).is_cuda

    outputs = []
    encoder_inputs, encoder_lengths = var
    encoder_inputs = cuda(encoder_inputs, use_cuda)
    # encoder_lengths = cuda(encoder_lengths, use_cuda)
    encoder_unpacked, encoder_hidden = encoder(encoder_inputs, encoder_lengths, return_packed=False)

    decoder_hidden = encoder_hidden
    decoder_inputs, decoder_lenghts = trg_field.numericalize(([[SOS]], [1]), device=-1)
    decoder_inputs = cuda(decoder_inputs, use_cuda)
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
            ni = int(topi.cpu().numpy()[0][0][0])
            if trg_field.vocab.itos[ni] == EOS:
                outputs.append(EOS)
                break
            else:
                outputs.append(trg_field.vocab.itos[ni])
            decoder_inputs = Variable(torch.LongTensor([[ni]]))
            decoder_inputs = cuda(decoder_inputs, use_cuda)
        outputs = " ".join(outputs)
    return outputs.strip()

def sample(encoder, decoder, var, trg_field, max_len=30, greedy=False, config=None):
    """ Sample an output given the input
    Args:
        var: (Tensor, List) tuple

    Returns: (outputs, log_probas)
        outputs: a list of str
        log_probas: Tensor (1, len)

    """
    # use_cuda = next(encoder.parameters()).is_cuda
    use_cuda = config['use_cuda']
    ls = nn.LogSoftmax()
    log_probas = []
    outputs = []

    encoder_inputs, encoder_lengths = var
    encoder_inputs = cuda(encoder_inputs, use_cuda)
    encoder_unpacked, encoder_hidden = encoder(encoder_inputs, encoder_lengths, return_packed=False)
    decoder_hidden = encoder_hidden
    decoder_inputs, decoder_lenghts = trg_field.numericalize(([[SOS]], [1]), device=-1)
    decoder_inputs = cuda(decoder_inputs, use_cuda)
    for i in range(max_len):
        # TODO: shall we use eval mode?
        # decoder_unpacked: (1, 1, vocab_size), eval() is effective to Dropout and BatchNorm
        decoder_unpacked, decoder_hidden = decoder.eval()(decoder_inputs, decoder_hidden, encoder_unpacked, encoder_lengths)
        if greedy:
            logp, ni = torch.max(ls(decoder_unpacked.squeeze()), 0)
            # ni must be an integer, not like numpy.int32
            ni = int(ni.data.cpu().numpy()[0])
        else:
            m = Categorical(F.softmax(decoder_unpacked.squeeze()))
            ni = m.sample()
            logp = m.log_prob(ni)
            ni = int(ni.cpu().data.numpy()[0])
        if trg_field.vocab.itos[ni] == EOS:
            outputs.append(EOS)
            log_probas.append(logp)
            # Note that the log proba of EOS is not saved,
            # In this case, there will be no log proba
            break
        else:
            outputs.append(trg_field.vocab.itos[ni])
        log_probas.append(logp)
        decoder_inputs = Variable(torch.LongTensor([[ni]]))
        decoder_inputs = cuda(decoder_inputs, use_cuda)
    # => row vector
    seq_log_probas = torch.cat([p.unsqueeze(1) for p in log_probas], 1)
    return outputs, seq_log_probas



def random_eval(encoder, decoder, batch, n, src_field, trg_field, config=None,
        greedy=False, metric='rouge', logger=None):

    enc_inputs, enc_lengths = batch.src
    dec_inputs, dec_lengths = batch.trg

    N = enc_inputs.size()[1]
    idx = np.random.choice(N, n)
    for i in idx:
        logger.info('> ' + tostr(clean(itos(enc_inputs[:,i].cpu().data.numpy(), src_field))))
        logger.info('= ' + tostr(clean(itos(dec_inputs[:,i].cpu().data.numpy(), trg_field))))
        enc_input = (enc_inputs[:,i].unsqueeze(1), torch.LongTensor([enc_lengths[i]]))
        outputs, _ = sample(encoder, decoder, enc_input, trg_field, max_len=30, greedy=greedy, config=config)
        # sent = evaluate(encoder, decoder, enc_input, trg_field=trg_field, beam_size=beam_size)
        logger.info('< ' + tostr(clean(outputs)) + '\n')


def score(hyps, refs, metric='rouge'):
    """
    Args:
        hyp: predicted sentence
        ref: reference sentence
        metric: metric to use
    """
    assert metric in ['rouge', 'bleu']
    if metric is 'rouge':
        rouge = Rouge()
    # {"rouge-1": {"f": _, "p": _, "r": _}, "rouge-2" : { ..     }, "rouge-3": { ... }}
        scores = rouge.get_scores(hyps, refs, avg=True)
    elif metric is 'bleu':
        pass
    return scores

def get_rewards(encoder, decoder,src_field, trg_field, beam_size=-1, metric='rouge'):
    pass

def synchronize(config):
    if config['use_cuda']:
        torch.cuda.synchronize()

def clean(l):
    """
    Remove special symbols from a list of str
    """
    symbols = [EOS, SOS, PAD]
    return [w for w in l if w not in symbols]

def tostr(l):
    return " ".join(l)

def get_rouge(hyps, refs):
    """
    Get average ROUGE-1, ROUGE-2, ROUGE-L F-1 scores
    """
    scores = score(hyps=hyps, refs=refs, metric='rouge')
    s = "\nROUGE-1: {0}\nROUGE-2: {1}\nROUGE-L: {2}\n".format(
            scores['rouge-1']['f'], scores['rouge-2']['f'],
            scores['rouge-l']['f'])
    return s
