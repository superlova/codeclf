'''
Author: Zyt
Date: 2020-12-02 19:21:21
LastEditTime: 2020-12-20 09:04:26
LastEditors: superlova
Description: 从原始数据（corpus）构建带标签数据集（tf.data）。数据来源包含docstring和代码。初始标签由FSM决定。
FilePath: \codeclf\preprocessing\DataProcessor.py
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import sys, os
import ast

# project_dir = 'C:/Users/zyt/Documents/GitHub Repositories/codeclf_gui/codeclf'
# sys.path.append(project_dir)

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
        """将df['code']分行、获得上下文、转化为dataset并打乱"""
        codes, docs = self.context_encoder.context_merge_all(file_texts, before, after)
        df_codes = pd.DataFrame(data=codes, columns=['before', 'text', 'after', 'label'])
        df_docs = pd.DataFrame(data=docs, columns=['before', 'text', 'after', 'label'])

        print(f"df_codes:{len(df_codes)}, df_docs:{len(df_docs)}")
        df = pd.concat([df_codes, df_docs], ignore_index=True)

        label = df.pop('label')
        dataset = tf.data.Dataset.from_tensor_slices((df.values, label.values))
        dataset = dataset.shuffle(100000, reshuffle_each_iteration=False).repeat()
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

def test_tf_data():
    corpus_path = '../datasets/df_test_corpus.tar.bz2'
    df_data = pd.read_pickle(corpus_path)
    data = df_data['code']
    dp = DataProcessor()
    dataset = dp.process_context_tfdata_merge(data, before=1, after=1)
    labels = []
    for data, label in dataset.take(50):
        labels.append(label.numpy())
    print(labels)

def test_tf_data_model():
    import functools
    from utils.CodeTokenizer import ContextCodeTokenizer
    corpus_path = '../datasets/df_test_corpus.tar.bz2'
    df_data = pd.read_pickle(corpus_path)
    data = df_data['code']
    VALID_SIZE = len(data)


    dp = DataProcessor()
    dataset = dp.process_context_tfdata_merge(data, before=1, after=1)
    INPUT_LENGTH = 90
    BATCH_SIZE = 1024

    vocab_path = '../vocabs/nosplit_keyword_vocab50000.txt'
    tokenizer = ContextCodeTokenizer(vocab_path)
    def _feature_to_id_bta(features, label, tokenizer, input_length):
        """Context特有的方法，将dataset中的上下文结合到一起并生成ids"""
        return tokenizer.from_feature_to_token_id_bta(features[0].decode("utf-8"),
                                                      features[1].decode("utf-8"),
                                                      features[2].decode("utf-8"), maxlen=input_length), label
    def tf_feature_to_id(features, label):
        """Context特有的方法，将feature_to_id_bta包装为tf.py_function"""
        label_shape = label.shape
        [features, label] = tf.numpy_function(feature_to_id,
                                    inp=[features, label],
                                    Tout=[tf.int32, tf.int64])
        features.set_shape((INPUT_LENGTH,))
        label.set_shape(label_shape)
        return features, label

    feature_to_id = functools.partial(_feature_to_id_bta, tokenizer=tokenizer, input_length=INPUT_LENGTH)

    val_ds = dataset.map(tf_feature_to_id, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.models.load_model('../models/lstm_model_token_50000_context_try_1.hdf5')
    model.summary()

    validation_steps = tf.math.ceil(VALID_SIZE / BATCH_SIZE).numpy()
    print(validation_steps)

    res = model.predict(val_ds, steps=validation_steps)
    res = np.asarray(res)
    # print(res.round())
    print(np.squeeze(res.round()))
    # print(val_ds)


def main():
    logging.basicConfig(
        level=logging.INFO
    )
    # test_context()
    # test_tf_data()
    test_tf_data_model()


if __name__ == '__main__':
    main()
