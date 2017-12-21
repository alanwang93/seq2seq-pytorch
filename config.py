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

def translation():
    c = {}
    # filename is of form: 'train.src', 'test.trg'
    c['root'] = 'data/translation/'
    c['prefix'] = 'translation_'
    c['splits'] = ['train', 'test']
    # names of Spacy models
    c['src_lang'] = 'fr'
    c['trg_lang'] = 'en_core_web_sm'
    c['model_path'] = './models/'
    c['log_step'] = 50
    c['save_step'] = 500
    c['beam_size'] = -1
    # model settings
    c['encoder_embed_size'] = 256
    c['decoder_embed_size'] = 256
    c['share_embed'] = False
    c['encoder_hidden_size'] = 512
    c['decoder_hidden_size'] = 512
    # training settings
    c['num_epochs'] = 5
    c['batch_size'] = 64
    c['learning_rate'] = 0.001
    c['encoder_vocab'] = 10000
    c['decoder_vocab'] = 10000

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

def chatbot():
    # TODO
    pass
