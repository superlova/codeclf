'''
Author: Zyt
Date: 2020-12-02 19:21:21
LastEditTime: 2020-12-17 23:55:29
LastEditors: superlova
Description: 从原始数据（corpus）构建带标签数据集（tf.data）。数据来源包含docstring和代码。初始标签由FSM决定。
FilePath: \codeclf\preprocessing\DataProcessor.py
'''

import tensorflow as tf
import pandas as pd
import logging
import sys, os
import ast

project_dir = 'C:/Users/zyt/Documents/GitHub Repositories/codeclf_gui/codeclf'
sys.path.append(project_dir)

from preprocessing.ContextEncoder import ContextEncoder
from preprocessing.CorpusEncoder import CorpusEncoder
# from preprocessing.ASTEncoder import ASTEncoder


class DataProcessor(object):
    def __init__(self):
        self.context_encoder = ContextEncoder()
        self.corpus_encoder = CorpusEncoder()
        # self.ast_encoder = ASTEncoder()

    def process_context(self, corpus):
        context_data = self.context_encoder.context_encode(corpus, before=1, after=1)
        df = pd.DataFrame(data=context_data)
        print(df.head())

    # def process_files(self, file_texts, before=1, after=1):
    #     codes, docs = self.context_encoder.context_merge_all(file_texts, before, after)
    #     df_codes = pd.DataFrame(data=codes, columns=['before', 'text', 'after', 'label'])
    #     df_docs = pd.DataFrame(data=docs, columns=['before', 'text', 'after', 'label'])
    #
    #     print(f"df_codes:{len(df_codes)}, df_docs:{len(df_docs)}")
    #     df = pd.concat([df_codes, df_docs], ignore_index=True)
    #     return df

    def process(self, corpus):
        context_data = self.context_encoder.context_encode(corpus, before=1, after=1)
        corpus_tokens = self.corpus_encoder.get_context_only_id(corpus)
        # tree = ast.parse(corpus)
        # corpus_tree_preorder = self.ast_encoder.get_ast_preorder(tree)
        # corpus_tree_inorder = self.ast_encoder.get_ast_inorder(tree)

        print(context_data)
        print(corpus_tokens)

    def process_context_tfdata_divide(self, file_texts, before=1, after=1):
        """
        以列表的形式保存before和after，目前存在不能序列化的问题
        :param file_texts: 
        :param before: 
        :param after: 
        :return: 
        """
        codes, docs = self.context_encoder.context_divide_all(file_texts, before, after)
        df_codes = pd.DataFrame(data=codes, columns=['before', 'text', 'after', 'label'])
        df_docs = pd.DataFrame(data=docs, columns=['before', 'text', 'after', 'label'])

        print(f"df_codes:{len(df_codes)}, df_docs:{len(df_docs)}")
        df = pd.concat([df_codes, df_docs], ignore_index=True)

        label = df.pop('label')
        dataset = tf.data.Dataset.from_tensor_slices((df.values, label.values))
        return dataset

    def process_context_tfdata_merge(self, file_texts, before=1, after=1):

        codes, docs = self.context_encoder.context_merge_all(file_texts, before, after)
        df_codes = pd.DataFrame(data=codes, columns=['before', 'text', 'after', 'label'])
        df_docs = pd.DataFrame(data=docs, columns=['before', 'text', 'after', 'label'])

        print(f"df_codes:{len(df_codes)}, df_docs:{len(df_docs)}")
        df = pd.concat([df_codes, df_docs], ignore_index=True)

        label = df.pop('label')
        dataset = tf.data.Dataset.from_tensor_slices((df.values, label.values))
        return dataset


def test_process():
    corpus_path = '../datasets/df_test_corpus.tar.bz2'
    df_data = pd.read_pickle(corpus_path)
    data = df_data['code'][90]

    dp = DataProcessor()
    dp.process(data)

def test_context():
    corpus_path = '../datasets/df_test_corpus.tar.bz2'
    df_data = pd.read_pickle(corpus_path)
    data = df_data['code']

    dp = DataProcessor()
    df_test = dp.process_files(data)
    print(df_test.tail())

def test_tfdata():
    corpus_path = '../datasets/df_test_corpus.tar.bz2'
    df_data = pd.read_pickle(corpus_path)
    data = df_data['code']

    dp = DataProcessor()
    dataset = dp.process_context_tfdata_divide(data, before=2, after=2)
    for f, t in dataset.take(5):
        print(f, t)


def main():
    logging.basicConfig(
        level=logging.INFO
    )
    # test_context()
    test_tfdata()


if __name__ == '__main__':
    main()
