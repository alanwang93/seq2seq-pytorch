#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Yifan WANG <yifanwang1993@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Configurations for different tasks
"""
from torchtext.data import Example

def gigawords():
    c = {}
    # filename is of form: 'train.src', 'test.trg'
    c['root'] = 'data/summarization/'
    c['prefix'] = 'summarization'
    c['splits'] = ['train', 'test', 'valid']
    # names of Spacy models
    c['src_lang'] = 'en_core_web_sm'
    c['trg_lang'] = 'en_core_web_sm'
    c['model_path'] = './models/'
    c['log_step'] = 1000
    c['save_step'] = 1000
    c['test_step'] = 4000
    c['beam_size'] = -1
    # model settings
    c['encoder_embed_size'] = 300
    c['decoder_embed_size'] = 300
    c['share_embed'] = False
    c['encoder_hidden_size'] = 512
    c['decoder_hidden_size'] = 512
    # training settings
    c['num_epoch'] = 5
    c['max_epoch'] = 50
    c['num_layers'] = 1
    c['batch_size'] = 32
    c['learning_rate'] = 0.0001
    c['encoder_vocab'] = 30000
    c['decoder_vocab'] = 20000

    def load(src_path, trg_path, src_field, trg_field):
        """
        Function used to load examples from file
        """
        src = open(src_path, 'r').readlines()
        trg = open(trg_path, 'r').readlines()
        examples = []
        for (l1, l2) in zip(src,trg):
            examples.append(Example.fromlist([l1, l2], [('src', src_field), ('trg', trg_field)]))
        return examples

    c['load'] = load
    return c
