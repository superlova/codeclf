#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/23 15:50
# @Author  : Zyt
# @Site    : 
# @File    : test_discover_dataset_0323.py
# @Software: PyCharm

import pandas as pd

def standardize_df(df):
    df_codes = pd.concat([df['code'], pd.Series([0 for i in range(len(df))])], axis=1)
    df_codes.columns = ["data", "label"]
    df_docs = pd.concat([df['docstring'], pd.Series([1 for i in range(len(df))])], axis=1)
    df_docs.columns = ["data", "label"]
    return pd.concat([df_codes, df_docs], ignore_index=True)

df = pd.read_pickle("datasets/df_train_corpus.tar.bz2")
df_new = standardize_df(df)
df_new.to_pickle("datasets/df_train_corpus_standardized.tar.bz2")


df = pd.read_pickle("datasets/df_test_corpus.tar.bz2")
df_new = standardize_df(df)
df_new.to_pickle("datasets/df_test_corpus_standardized.tar.bz2")


df = pd.read_pickle("datasets/df_valid_corpus.tar.bz2")
df_new = standardize_df(df)
df_new.to_pickle("datasets/df_valid_corpus_standardized.tar.bz2")
