# -*- coding: utf-8 -*-
# @Author  : superlova
# @Time    : 2020/11/10 11:39
# @Function: 将原始数据集转化为df_corpus

import os
import json

import pandas as pd

def load_data_to_df(in_path):
    # input
    results = []
    path = os.path.join(in_path)
    with open(path, 'r') as file:
        for line in file:
            results.append(json.loads(line))
    # output
    df = pd.DataFrame(results)
    df = df[['code', 'docstring']]
    return df


def main():
    # 指定训练集存放目录
    train_path = '../datasets/python/python/final/jsonl/train'
    test_path = '../datasets/python/python/final/jsonl/test'
    valid_path = '../datasets/python/python/final/jsonl/valid'

    test_file = os.path.join(test_path, 'python_test_0.jsonl')
    valid_file = os.path.join(valid_path, 'python_valid_0.jsonl')

    # 将这一步的结果保存
    df_train_corpus = load_data_to_df(os.path.join(train_path, 'python_train_0.jsonl'))
    for i in range(1, 14):
        in_path = os.path.join(train_path, 'python_train_{}.jsonl'.format(i))
        df_train_corpus = pd.concat([df_train_corpus, load_data_to_df(in_path)], ignore_index=True)

    df_test_corpus = load_data_to_df(test_file)
    df_valid_corpus = load_data_to_df(valid_file)

    saved_path = '../datasets'
    df_train_corpus.to_pickle(os.path.join(saved_path, 'df_train_corpus.tar.bz2'), protocol=4)
    df_test_corpus.to_pickle(os.path.join(saved_path, 'df_test_corpus.tar.bz2'), protocol=4)
    df_valid_corpus.to_pickle(os.path.join(saved_path, 'df_valid_corpus.tar.bz2'), protocol=4)


if __name__ == '__main__':
    main()
