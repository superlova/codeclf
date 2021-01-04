# -*- coding: utf-8 -*-
# @Author  : superlova
# @Time    : 2020/11/1 18:29
# @Function:
import ast
import pandas as pd

def check_ast(item):
    try:
        ast.parse(item)
        return True
    except SyntaxError:
        return False

def main():
    df_train = pd.read_pickle('../datasets/df_train_corpus.tar.bz2')
    df_valid = pd.read_pickle('../datasets/df_valid_corpus.tar.bz2')
    df_test = pd.read_pickle('../datasets/df_test_corpus.tar.bz2')

    df_train_pass = df_train[df_train['code'].apply(check_ast)]
    df_valid_pass = df_valid[df_valid['code'].apply(check_ast)]
    df_test_pass = df_test[df_test['code'].apply(check_ast)]

    print(f'df_train:{len(df_train)}, df_valid:{len(df_valid)}, df_test:{len(df_test)}')
    print(f'df_train_pass:{len(df_train_pass)}, df_valid_pass:{len(df_valid_pass)}, df_test_pass:{len(df_test_pass)}')


if __name__ == '__main__':
    main()
    pd.read_pickle()