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
import os, time, sys, datetime, argparse, pickle, json

from model import EncoderRNN, DecoderRNN
import config
from utils import *

EOS = "<eos>"
SOS = "<sos>"
PAD = "<pad>"
np.random.seed(666)


def main(args):
    # Load configurations
    start = time.time()
    c = getattr(config, args.config)()
    c['use_cuda'], c['exp'], c['mode'] = args.use_cuda, args.exp, args.mode
    assert c['exp'] is not None, "'exp' must be specified."
    logger  = init_logging('log/{0}_{1}_{2}.log'.format(c['prefix'], c['exp'], start))
    logger.info(since(start) + "Loading data with configuration '{0}':\n{1}".format(args.config, str(c)))

    # Load datasets
    datasets, src_field, trg_field = load_data(c)

    train = datasets['train']
    n_train = len(train.examples)
    test = datasets['test']
    n_test = len(test.examples)
    valid = datasets['valid']
    n_valid = len(valid.examples)
    num_epoch = c['num_epoch'] if not args.early_stopping else c['max_epoch']
    batch_per_epoch = n_train // c['batch_size'] if n_train % c['batch_size'] == 0 else n_train // c['batch_size']+1
    n_iters = batch_per_epoch * num_epoch

    # Build vocabularies
    src_field.build_vocab(train, max_size=c['encoder_vocab'])
    trg_field.build_vocab(train, max_size=c['decoder_vocab'])
    PAD_IDX = trg_field.vocab.stoi[PAD] # default=1

    logger.info("Source vocab: {0}".format(len(src_field.vocab.itos)))
    logger.info("Target vocab: {0}".format(len(trg_field.vocab.itos)))
    logger.info(since(start) + "{0} training samples, {1} epochs, batch size={2}, {3} batches per epoch.".format(
            n_train, num_epoch, c['batch_size'], batch_per_epoch))

    train_iter = iter(BucketIterator(
        dataset=train, batch_size=c['batch_size'], sort=True,
        sort_key=lambda x: len(x.src), device=-1))

    test_iter = iter(BucketIterator(
        dataset=test, batch_size=1, sort=True,
        sort_key=lambda x: len(x.src), device=-1))

    valid_iter = iter(BucketIterator(
        dataset=valid, batch_size=1,sort=True,
        sort_key=lambda x: len(x.src), device=-1))

    del train
    del test
    del valid

    encoder = EncoderRNN(vocab_size=len(src_field.vocab), embed_size=c['encoder_embed_size'],\
            hidden_size=c['encoder_hidden_size'], padding_idx=PAD_IDX, n_layers=c['num_layers'])
    decoder = DecoderRNN(vocab_size=len(trg_field.vocab), embed_size=c['decoder_embed_size'],\
            hidden_size=c['decoder_hidden_size'], encoder_hidden=c['encoder_hidden_size'],\
            padding_idx=PAD_IDX, n_layers=c['num_layers'])
    if not args.resume:
        # Train from scratch
        params = list(encoder.parameters()) +  list(decoder.parameters())
        optimizer = optim.Adam(params, lr=c['learning_rate'])
        init_epoch = init_step = 0
        history = {'epochs':[],
                'train_loss':[],
                'valid_loss':[],
                'test_loss':[],
                'test_score':[],
                'best_epoch':-1,
                'best_loss':float("inf")}

        logger.info(since(start) + "Start training... {0} epochs, {1} steps per epoch.".format(
                num_epoch, batch_per_epoch))
    else:
        assert os.path.isfile("{0}{1}_{2}.pkl".format(c['model_path'], c['prefix'], c['exp']))
        # Load checkpoint
        logger.info(since(start) + "Loading from {0}{1}_{2}.pkl".format(c['model_path'], c['prefix'], c['exp']))
        cp = torch.load("{0}{1}_{2}.pkl".format(c['model_path'], c['prefix'], c['exp']))
        encoder.load_state_dict(cp['encoder'])
        decoder.load_state_dict(cp['decoder'])
        params = list(encoder.parameters()) +  list(decoder.parameters())
        optimizer = optim.Adam(params, lr=c['learning_rate'])
        optimizer.load_state_dict(cp['optimizer'])
        init_epoch, init_step, others, history = cp['epoch'], cp['step'], cp['others'], cp['history']
        del cp
        logger.info(since(start) + "Resume training from {0}/{1} epoch, {2}/{3} step".format(
                init_epoch+1, num_epoch, init_step+1, batch_per_epoch))

    if c['use_cuda']:
        encoder.cuda()
        decoder.cuda()
    else:
        encoder.cpu()
        decoder.cpu()

    CEL = nn.CrossEntropyLoss(size_average=True, ignore_index=PAD_IDX)
    print_loss = 0

    # Start training
    for e in range(init_epoch, num_epoch):
        for j in range(init_step, batch_per_epoch):
            init_step = 0
            i = batch_per_epoch*e + j + 1 # global step

            batch = next(train_iter)
            encoder_inputs, encoder_lengths = batch.src
            decoder_inputs, decoder_lengths = batch.trg

            encoder_inputs = cuda(encoder_inputs, c['use_cuda'])
            decoder_inputs = cuda(decoder_inputs, c['use_cuda'])

            encoder_unpacked, encoder_hidden = encoder(encoder_inputs, encoder_lengths, return_packed=False)
            # we don't remove the last symbol
            decoder_unpacked, decoder_hidden = decoder(decoder_inputs[:-1,:], encoder_hidden, encoder_unpacked, encoder_lengths)
            trg_len, batch_size, d = decoder_unpacked.size()
            # remove first symbol <SOS>
            ce_loss = CEL(decoder_unpacked.view(trg_len*batch_size, d), decoder_inputs[1:,:].view(-1))
            print_loss += ce_loss.data

            # Self-critical sequence training
            assert args.self_critical >= 0. and args.self_critical <= 1.
            if args.self_critical > 1e-5:
                sc_loss = cuda(Variable(torch.Tensor([0.])), c['use_cuda'])
                for j in range(batch_size):
                    enc_input = (encoder_inputs[:,j].unsqueeze(1), torch.LongTensor([encoder_lengths[j]]))
                    # use self critical training
                    greedy_out, _ = sample(encoder, decoder, enc_input, trg_field,
                            max_len=30, greedy=True, config=c)
                    greedy_sent = tostr(clean(greedy_out))
                    sample_out, sample_logp = sample(encoder, decoder, enc_input, trg_field,
                            max_len=30, greedy=False, config=c)
                    sample_sent = tostr(clean(sample_out))
                    # Ground truth
                    gt_sent = tostr(clean(itos(decoder_inputs[:,j].cpu().data.numpy(), trg_field)))
                    greedy_score = score(hyps=greedy_sent, refs=gt_sent, metric='rouge')
                    sample_score = score(hyps=sample_sent, refs=gt_sent, metric='rouge')
                    reward = Variable(torch.Tensor([sample_score["rouge-1"]['f'] - greedy_score["rouge-1"]['f']]), requires_grad=False)
                    reward = cuda(reward, c['use_cuda'])
                    sc_loss -= reward*torch.sum(sample_logp)

                if i % c['log_step'] == 0:
                    logger.info("CE loss: {0}".format(ce_loss))
                    logger.info("RL loss: {0}".format(sc_loss))
                    logger.info("Ground truth: {0}".format(gt_sent))
                    logger.info("Greedy: {0}, {1}".format(greedy_score['rouge-1']['f'], greedy_sent))
                    logger.info("Sample: {0}, {1}".format(sample_score['rouge-1']['f'], sample_sent))

                loss = (1-args.self_critical) * ce_loss + args.self_critical * sc_loss
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del encoder_inputs, decoder_inputs

            if i % c['save_step'] == 0:
                # Save model for resuming
                synchronize(c)
                logger.info(since(start) + "Saving models.")
                cp = {'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(), 'others': {},
                    'step': j+1, 'epoch': e, 'history': history}
                torch.save(cp, "{0}{1}_{2}.pkl".format(c['model_path'], c['prefix'], c['exp']))

            if i % c['log_step'] == 0:
                synchronize(c)
                logger.info(since(start) + 'epoch {0}/{1}, iteration {2}/{3}'.format(e, num_epoch, i, batch_per_epoch))
                logger.info("\tTrain loss: {0}".format(print_loss.cpu().numpy().tolist()[0] / c['log_step']))
                print_loss = 0
                random_eval(encoder, decoder, batch, n=1, src_field=src_field, trg_field=trg_field, config=c,
                        greedy=True, logger=logger)

            # Evaluate on test set
            if i % c['test_step'] == 0:
                test_loss = 0
                test_rouge = 0
                refs = []
                greedys = []
                for j in range(n_test):
                    test_batch = next(test_iter)
                    test_encoder_inputs, test_encoder_lengths = test_batch.src
                    test_decoder_inputs, test_decoder_lengths = test_batch.trg
                    # GPU
                    test_encoder_inputs = cuda(Variable(test_encoder_inputs.data, volatile=True), c['use_cuda'])
                    test_decoder_inputs = cuda(Variable(test_decoder_inputs.data, volatile=True), c['use_cuda'])

                    test_encoder_unpacked, test_encoder_hidden = encoder(test_encoder_inputs, test_encoder_lengths, return_packed=False)
                    # we don't remove the last symbol
                    test_decoder_unpacked, test_decoder_hidden = decoder(test_decoder_inputs[:-1,:], test_encoder_hidden, test_encoder_unpacked, test_encoder_lengths)
                    trg_len, batch_size, d = test_decoder_unpacked.size()
                    # remove first symbol <SOS>
                    test_ce_loss = CEL(test_decoder_unpacked.view(trg_len*batch_size, d), test_decoder_inputs[1:,:].view(-1))
                    test_loss += test_ce_loss.data

                    test_enc_input = (test_encoder_inputs[:,0].unsqueeze(1), torch.LongTensor([test_encoder_lengths[0]]))
                    test_greedy_out, _ = sample(encoder, decoder, test_enc_input, trg_field,
                            max_len=30, greedy=True, config=c)
                    test_greedy_sent = tostr(clean(test_greedy_out))

                    test_gt_sent = tostr(clean(itos(test_decoder_inputs[:,0].cpu().data.numpy(), trg_field)))
                    refs.append(test_gt_sent)
                    greedys.append(test_greedy_sent)


                rouges = get_rouge(hyps=greedys, refs=refs)
                synchronize(c)
                logger.info(since(start) + "Test loss: {0}".format(test_loss.cpu().numpy().tolist()[0]/n_test))
                logger.info(rouges)

        # One epoch is finished
        logger.info(since(start) + "Epoch {0} is finished.".format(e))
        # Evaluate on validation set and perform early stopping
        valid_loss = 0
        for j in range(n_valid):
            batch = next(valid_iter)
            encoder_inputs, encoder_lengths = batch.src
            decoder_inputs, decoder_lengths = batch.trg

            encoder_inputs = cuda(encoder_inputs, c['use_cuda'])
            decoder_inputs = cuda(decoder_inputs, c['use_cuda'])

            encoder_unpacked, encoder_hidden = encoder(encoder_inputs, encoder_lengths, return_packed=False)
            decoder_unpacked, decoder_hidden = decoder(decoder_inputs[:-1,:], encoder_hidden, encoder_unpacked, encoder_lengths)
            trg_len, batch_size, d = decoder_unpacked.size()
            valid_ce = CEL(decoder_unpacked.view(trg_len*batch_size, d), decoder_inputs[1:,:].view(-1))
            valid_loss += valid_ce.data
        history['valid_loss'].append(valid_loss.cpu().numpy().tolist()[0]/n_valid)
        synchronize(c)
        logger.info(since(start) + "Saving models.")
        cp = {'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(), 'others': {},
            'step': j+1, 'epoch': e, 'history': history}
        torch.save(cp, "{0}{1}_{2}_epoch{3}.pkl".format(c['model_path'], c['prefix'], c['exp'], e))
        logger.info("Epoch {0}, valid loss: {1}".format(e, history['valid_loss'][-1]))
        if history['valid_loss'][-1] < history['best_loss']:
            history['best_loss'] = history['valid_loss'][-1] 
            history['best_epoch'] = e
            torch.save(cp, "{0}{1}_{2}_best.pkl".format(c['model_path'], c['prefix'], c['exp']))
        elif args.early_stopping and e - history['best_epoch'] > patient:
            # early stopping
            logger.info(since(start) + "Early stopping at epoch {0}, best result at epoch {1}".format(e, history['best_epoch']))
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None ,
                        help='model configurations, defined in config.py')
    parser.add_argument('--disable_cuda', type=bool, default=False)
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--early_stopping', dest='early_stopping', action='store_true',
            help='With early stopping, the training will end when valid loss doesn\'t decrease \
            for `--patient` epochs, or at `max_epoch` epoch.')
    parser.add_argument('--self_critical', type=float, default=0.)
    parser.add_argument('--exp', type=str, default=None, help='A string that specify the name of the experiment')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--patient', type=int, default=5)
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda and torch.cuda.is_available()

    print(args)
    go = input('Start? y/n\n')
    if go != 'y':
        exit()
    if args.use_cuda:
        print("Use GPU...")
    else:
        print("Use CPU...")
    main(args)
