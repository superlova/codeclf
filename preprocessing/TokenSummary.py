# -*- coding: utf-8 -*-
# @Author  : superlova
# @Time    : 2020/11/1 18:29
# @Function:

import tokenize
from collections import Counter
import pandas as pd

from utils.Utils import split_token
from utils.Utils import create_generator, timethis
from utils.Utils import iskeyword

@timethis
def make_vocab_split_keyword(data, limit):

    generator_train = create_generator(data)
    tokens = tokenize.tokenize(generator_train)
    result = []
    try:
        for token_type, token_content, _, _, _ in tokens:
            if token_type == tokenize.NAME and iskeyword(token_content):  # keyword
                result.append((5, token_content))
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
                result.append((5, token_content))
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


def main():
    df_train = pd.read_pickle('../datasets/df_train_line.tar.bz2')
    make_vocab_split_keyword(df_train['data'], 10000)
    # make_vocab_nosplit_keyword(df_train['data'], 50000)
    # make_vocab_only_keyword(df_train['data'])


if __name__ == '__main__':
    main()
