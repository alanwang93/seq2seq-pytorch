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

def split_data(root, filenames, exts, train_ratio=0.8, test_ratio=0.2):
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
