#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/23 15:37
# @Author  : Zyt
# @Site    : 
# @File    : test_train_0323.py
# @Software: PyCharm

import os, sys
# project_dir = os.path.join(os.getcwd(), 'comment-checker')
# sys.path.append(project_dir)
project_dir = os.getcwd()

from utils.CodeTokenizer import CodeTokenizer, CodeSplitTokenizer
from utils.CodeTokenizer import ContextCodeTokenizer, ContextCodeSplitTokenizer
from utils.Utils import timethis
from preprocessing.DataProcessor import DataProcessor
from train.ClfModel import ContextModel, ContextSpiltModel

import numpy as np
import pandas as pd
import logging
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO
)

#?
model_type = 'bilstm_1'
context_before = 1
context_after = 1
context_mode = 'bta'
data_num = 0.001
vocab_size = 10000
try_time = 7
model_name = f'{model_type}_{context_mode}{context_before}{context_after}data{data_num}vocab{vocab_size}split_{try_time}'
print(model_name)


checkpoint_path = os.path.join(project_dir, 'checkpoint/{}'.format(model_name))
print('checkpoint_path: ', checkpoint_path)

vocab_path = os.path.join(project_dir, 'vocabs/nosplit_keyword_vocab50000.txt')
print('vocab_path: ', vocab_path)

train_path = os.path.join(project_dir, 'datasets/df_train_corpus.tar.bz2')
print('train_path: ', train_path)

valid_path = os.path.join(project_dir, 'datasets/df_valid_corpus.tar.bz2')
print('valid_path: ', valid_path)

test_path = os.path.join(project_dir, 'datasets/df_test_corpus.tar.bz2')
print('test_path: ', test_path)

save_path = os.path.join(project_dir, 'models/{}.hdf5'.format(model_name))
print('save_path: ', save_path)


trainer = ContextModel(before=context_before, after=context_after, context_mode=context_mode, batch_size=32)
trainer.load_vocab(vocab_path)

logging.info('datasets ready!')
trainer.construct_model(model_type=model_type)

logging.info('loading training data...')
trainer.load_datasets(train_path=train_path,
            valid_path=valid_path,
            frac=data_num)
logging.info('datasets ready!')