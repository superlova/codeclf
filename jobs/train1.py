
import os, sys
project_dir = os.path.abspath('..')
print(project_dir)
# project_dir = os.path.join(os.getcwd(), 'comment-checker')
sys.path.append(project_dir)

from utils.CodeTokenizer import CodeTokenizer, CodeSplitTokenizer
from utils.CodeTokenizer import ContextCodeTokenizer, ContextCodeSplitTokenizer
from utils.Utils import timethis
from preprocessing.DataProcessor import DataProcessor
from train.ClfModel import ContextModel, ContextSpiltModel

import numpy as np
import pandas as pd
import logging
import tensorflow as tf

from argparse import ArgumentParser


def train(model_type, context_mode, context_before, context_after, data_num, vocab_size, try_time, split):
    model_name = f'{model_type}_{context_mode}{context_before}{context_after}data{data_num}vocab{vocab_size}{split}_{try_time}'
    logging.info(model_name)

    checkpoint_path = os.path.join(project_dir, 'checkpoint/{}'.format(model_name))
    logging.info('checkpoint_path: {}'.format(checkpoint_path))

    if split == 'split':
        vocab_path = os.path.join(project_dir, 'vocabs/split_keyword_vocab50000.txt')
    else:
        vocab_path = os.path.join(project_dir, 'vocabs/nosplit_keyword_vocab50000.txt')
    logging.info('vocab_path: {}'.format(vocab_path))

    train_path = os.path.join(project_dir, 'datasets/df_train_corpus.tar.bz2')
    logging.info('train_path: {}'.format(train_path))

    valid_path = os.path.join(project_dir, 'datasets/df_valid_corpus.tar.bz2')
    logging.info('valid_path: {}'.format(valid_path))

    test_path = os.path.join(project_dir, 'datasets/df_test_corpus.tar.bz2')
    logging.info('test_path: {}'.format(test_path))

    save_path = os.path.join(project_dir, 'models/{}.hdf5'.format(model_name))
    logging.info('save_path: {}'.format(save_path))

    if split == 'split':
        trainer = ContextSpiltModel(before=context_before, after=context_after, context_mode=context_mode, batch_size=4096)
    else:
        trainer = ContextModel(before=context_before, after=context_after, context_mode=context_mode, batch_size=4096)

    trainer.load_vocab(vocab_path)

    logging.info('datasets ready!')
    trainer.construct_model(model_type=model_type)

    logging.info('loading training data...')
    trainer.load_datasets(train_path=train_path,
                          valid_path=valid_path,
                          frac=data_num)
    logging.info('datasets ready!')

    logging.info('start training...')
    trainer.train_model(checkpoint_save_path=checkpoint_path)

    logging.info('saving model...')
    trainer.save_model(save_path)
    trainer.plot_history(model_name)

    logging.info('evaluating model...')
    trainer.evaluate(test_path=test_path)


def main():
    logging.basicConfig(
        filename='train1.log',
        level=logging.INFO
    )
    parser = ArgumentParser(description='train')

    parser.add_argument('-mt', '--model_type',
                        dest='model_type',
                        metavar='model_type')

    parser.add_argument('-m', '--context_mode',
                        dest='context_mode',
                        metavar='context_mode')

    parser.add_argument('-cb', '--context_before',
                        dest='context_before',
                        metavar='context_before',
                        action='store',
                        type=int,
                        default=1)

    parser.add_argument('-ca', '--context_after',
                        metavar='context_after',
                        action='store',
                        type=int,
                        default=1,
                        dest='context_after')

    parser.add_argument('-d', '--data_num',
                        metavar='data_num',
                        action='store',
                        type=float,
                        default=1.0,
                        dest='data_num')

    parser.add_argument('-v', '--vocab_size',
                        metavar='vocab_size',
                        action='store',
                        type=int,
                        default=50000,
                        dest='vocab_size')

    parser.add_argument('-t', '--try_time',
                        metavar='try_time',
                        action='store',
                        type=int,
                        dest='try_time')

    parser.add_argument('-s', '--split',
                        action='store',
                        default='',
                        dest='split')

    args = parser.parse_args()

    args = {'model_type': args.model_type,
            'context_mode': args.context_mode,
            'context_before': args.context_before,
            'context_after': args.context_after,
            'data_num': args.data_num,
            'vocab_size': args.vocab_size,
            'try_time': args.try_time,
            'split': args.split
            }

    train(**args)


if __name__ == '__main__':
    main()