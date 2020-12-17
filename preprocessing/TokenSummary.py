'''
Date: 2020-12-02 11:10:04
LastEditors: superlova
LastEditTime: 2020-12-17 23:57:39
Description: 从训练集构建字典
FilePath: \codeclf\preprocessing\TokenSummary.py
'''
# -*- coding: utf-8 -*-
# @Author  : superlova
# @Time    : 2020/11/1 18:29
# @Function: 从训练集构建字典

import tokenize
from collections import Counter
import pandas as pd
import numpy as np

import os, sys
project_dir = 'C:/Users/zyt/Documents/GitHub Repositories/codeclf_gui/codeclf'
sys.path.append(project_dir)

from utils.Utils import split_token
from utils.Utils import create_generator, timethis
from utils.Utils import iskeyword

import logging

KEYWORD = 5

@timethis
def make_vocab_split_keyword(data, limit):

    generator_train = create_generator(data)
    tokens = tokenize.tokenize(generator_train)
    result = []
    try:
        for token_type, token_content, _, _, _ in tokens:
            if token_type == tokenize.NAME and iskeyword(token_content):  # keyword
                result.append((KEYWORD, token_content))
            elif token_type == tokenize.NAME and token_content.isascii():  # not keyword ascii
                sub_tokens = split_token(token_content) # 长单词分割
                for sub_token in sub_tokens:
                    result.append((token_type, sub_token))
            elif token_type == tokenize.OP:
                result.append((token_type, token_content))
    except tokenize.TokenError as e:
        print(e)

    save_summary_to_vocab(result, "split_keyword", limit)


@timethis
def make_vocab_nosplit_keyword(data, limit):

    generator_train = create_generator(data)
    tokens = tokenize.tokenize(generator_train)
    result = []
    try:
        for token_type, token_content, _, _, _ in tokens:
            if token_type == tokenize.NAME and iskeyword(token_content):  # keyword
                result.append((KEYWORD, token_content))
            elif token_type == tokenize.NAME and token_content.isascii() or token_type == tokenize.OP:  # not keyword ascii
                result.append((token_type, token_content))
    except tokenize.TokenError as e:
        print(e)

    save_summary_to_vocab(result, "nosplit_keyword", limit)


@timethis
def make_vocab_only_keyword(data):
    generator_train = create_generator(data)
    tokens = tokenize.tokenize(generator_train)
    result = []
    try:
        for token_type, token_content, _, _, _ in tokens:
            if token_type == tokenize.NAME and iskeyword(token_content) or token_type == tokenize.OP:  # keyword
                result.append((token_type, token_content))
    except tokenize.TokenError as e:
        print(e)

    save_summary_to_vocab(result, "only_keyword", -1)


@timethis
def make_vocab_split_simple(data):
    """输入按行分割的代码数据集，输出单词表。
    和以前的不同是将三引号换成了单引号（防止三引号直接被词法分析器忽略），
    以及去掉了开头的井号（防止一整行变成comment）
    必须保证data是按行输入的
    :param data:
    :return:
    """

    def data_generator(data):
        for row in data:
            temp = row.lstrip(" #'\"").rstrip(" '\"").replace('"""', '"').replace("'''", "'")
            if len(temp) >= 1:
                yield temp

    data = data_generator(data)

    generator_train = create_generator(data)
    tokens = tokenize.tokenize(generator_train)
    result = []
    try:
        for i, (token_type, token_content, _, _, _) in enumerate(tokens):
            if token_type == tokenize.NAME and iskeyword(token_content):  # keyword
                result.append((KEYWORD, token_content))
            elif token_type == tokenize.NAME and token_content.isascii():  # not keyword ascii
                sub_tokens = split_token(token_content) # 长单词分割
                for sub_token in sub_tokens:
                    result.append((token_type, sub_token))
            elif token_type == tokenize.OP:
                result.append((token_type, token_content))
    except tokenize.TokenError as e:
        print(e)

    return result


def save_summary_to_csv(result, mode, limit):
    df_tokens = pd.DataFrame(result, columns=['token_type', 'token_content'])
    counter = Counter(df_tokens['token_content'])
    vocab_all = pd.DataFrame(data=counter.most_common(), columns=['token', 'count'])
    vocab_all[:limit].to_csv(f"../vocabs/{mode}_vocab{limit}.csv")


def save_summary_to_vocab(result, mode, limit):
    df_tokens = pd.DataFrame(result, columns=['token_type', 'token_content'])
    counter = Counter(df_tokens['token_content'])
    vocab_all = pd.DataFrame(data=counter.most_common(), columns=['token', 'count'])
    with open(f"../vocabs/{mode}_vocab{limit}.txt", 'w') as f:
        for token in vocab_all['token'][:limit]:
            f.write(str(token) + '\n')


########################

def test_make_vocab():
    df_train = pd.read_pickle('../datasets/df_train_line.tar.bz2')
    make_vocab_split_keyword(df_train['data'], 10000)
    # make_vocab_nosplit_keyword(df_train['data'], 50000)
    # make_vocab_only_keyword(df_train['data'])


def test_vocab_simple():
    df_train = pd.read_pickle(os.path.join(project_dir, 'datasets/df_train_line.tar.bz2'))
    print(df_train)
    result = make_vocab_split_simple(df_train['data'])
    print(len(result))
    save_summary_to_vocab(result, "split_simple", limit=50000)


def test_data_generator():
    df_train = pd.read_pickle(os.path.join(project_dir, 'datasets/df_train_line.tar.bz2'))
    data_gen = data_generator(df_train['data'])

    i = 0
    count = 10
    for row in data_gen:
        if i < count:
            print(row)
        i += 1



def main():
    logging.basicConfig(
        level=logging.DEBUG
    )
    test_vocab_simple()
    # test_data_generator()
    # test_multiprocessing_simple_vocab()


if __name__ == '__main__':
    main()
