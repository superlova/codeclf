# -*- coding: utf-8 -*-
# @Author  : superlova
# @Time    : 2020/11/9 9:03
# @Function:

import tokenize
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import logging


def create_generator(data):
    """字符流生成器，在内部构建了一个闭包。
    为了节约内存，避免一次性加载文件内容"""

    def generator():
        for elem in data:
            try:
                yield str.encode(elem)
            except GeneratorExit:
                logging.info(f"GeneratorExit! EOF. {elem}")
                return
            except Exception as e:
                logging.info("Exception!" + str(type(e)), str(e))
                yield str.encode('')

    g = generator()  # 生成器

    def next_element():
        return next(g)

    return next_element  # 迭代器


class CharacterModel(object):
    def __init__(self):
        self.EMBEDDING_DIM = 50
        self.BATCH_SIZE = 32
        self.EPOCHS = 15
        self.MAXLEN = 70

        # define the raw dataset
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890~!@#$%^&*()` ,./<>?;':\"[]{}=-+_\t\r\n|\\"
        # create mapping of characters to integers (0-25) and the reverse
        self.char_to_int = dict((c, i + 2) for i, c in enumerate(self.alphabet))
        self.char_to_int['<pad>'] = 0
        self.char_to_int['<unk>'] = 1
        self.int_to_char = dict((i, c) for c, i in self.char_to_int.items())
        self.VOCAB_SIZE = len(self.char_to_int)

    def check_dict_char(self, word):
        try:
            return self.char_to_int[word]
        except Exception:
            return self.char_to_int['<unk>']

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def make_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(input_dim=self.VOCAB_SIZE, output_dim=self.EMBEDDING_DIM,
                                      input_length=self.MAXLEN, mask_zero=True),
            tf.keras.layers.LSTM(100),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def load_datasets(self, df_train, limit=-1):

        if limit != -1:
            df_train = df_train[:limit]

        df_train_code = df_train[df_train['label'] == 0]  # 代码
        df_train_doc = df_train[df_train['label'] == 1]  # Doc
        balanced_data_limit = min(len(df_train_code), len(df_train_doc))
        df_train_code = df_train_code[:balanced_data_limit]
        df_train_doc = df_train_doc[:balanced_data_limit]

        X = []
        y = []
        for data, label in zip(df_train_code['data'], df_train_code['label']):
            char_array = np.asarray(list(data), dtype=str)
            int_array = np.asarray(list(map(self.check_dict_char, char_array)))
            if len(int_array) >= 3:
                X.append(int_array)
                y.append(label)
        for data, label in zip(df_train_doc['data'], df_train_doc['label']):
            char_array = np.asarray(list(data), dtype=str)
            int_array = np.asarray(list(map(self.check_dict_char, char_array)))
            if len(int_array) >= 3:
                X.append(int_array)
                y.append(label)
        X = np.asarray(X)
        y = np.asarray(y)

        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)
        logging.info(f"len(X_train): {len(X_train)}, {len(self.y_train)}")
        self.X_train = pad_sequences(X_train, padding='post', value=0, maxlen=self.MAXLEN)
        self.X_test = pad_sequences(X_test, padding='post', value=0, maxlen=self.MAXLEN)

    def from_id_to_text(self, ids):
        tmp = ""
        for id in ids:
            if id >= 2:
                tmp += self.int_to_char[id]
            elif id == 1:
                tmp += self.int_to_char[1]
        return tmp.strip()

    def train_model(self):
        self.model.fit(self.X_train, self.y_train, batch_size=self.BATCH_SIZE,
                       epochs=self.EPOCHS, verbose=1, validation_data=(self.X_test, self.y_test))

    def save_model(self, save_path):
        self.model.save(save_path)
        
        
class TokenModel(object):
    def __init__(self, path):
        self.EMBEDDING_DIM = 200
        self.BATCH_SIZE = 32
        self.EPOCHS = 15
        self.MAXLEN = 30
        self.VOCAB_PATH = path
        self.make_switch_rule()

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def make_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(input_dim=self.VOCAB_SIZE, output_dim=self.EMBEDDING_DIM,
                                      input_length=self.MAXLEN, mask_zero=True),
            tf.keras.layers.LSTM(100),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(20),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def load_datasets(self, train_path, limit=-1):
        df_train = pd.read_pickle(train_path)
        if limit != -1:
            df_train = df_train[:limit]

        df_train_code = df_train[df_train['label'] == 0]  # 代码
        df_train_doc = df_train[df_train['label'] == 1]  # Doc
        balanced_data_limit = min(len(df_train_code), len(df_train_doc))
        df_train_code = df_train_code[:balanced_data_limit]
        df_train_doc = df_train_doc[:balanced_data_limit]

        X = []
        y = []
        for data, label in zip(df_train_code['data'], df_train_code['label']):
            int_array = np.asarray(self.from_text_to_id([data]))
            if len(int_array) >= 3:
                X.append(int_array)
                y.append(label)
        for data, label in zip(df_train_doc['data'], df_train_doc['label']):
            int_array = np.asarray(self.from_text_to_id([data]))
            if len(int_array) >= 3:
                X.append(int_array)
                y.append(label)
        X = np.asarray(X)
        y = np.asarray(y)

        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)
        logging.info(f"len(X_train): {len(X_train)}, {len(self.y_train)}")
        self.X_train = pad_sequences(X_train, padding='post', value=0, maxlen=self.MAXLEN)
        self.X_test = pad_sequences(X_test, padding='post', value=0, maxlen=self.MAXLEN)

    def load_vocab(self):
        df_vocab = pd.read_csv(self.VOCAB_PATH, index_col=0)
        self.token_2_id = {row['token']: index + 8 for index, row in df_vocab.iterrows()}
        self.token_2_id['<PAD>'] = 0
        self.token_2_id['<UNK>'] = 1
        self.token_2_id['<ID>'] = 2
        self.token_2_id['<PUN>'] = 3
        self.token_2_id['<NUM>'] = 4
        self.token_2_id['<STR>'] = 5
        self.token_2_id['<SPE>'] = 6
        self.token_2_id['<COM>'] = 7
        self.id_2_token = {v: k for k, v in self.token_2_id.items()}
        self.VOCAB_SIZE = len(self.token_2_id)
        logging.info(f"len(id_2_token): {len(self.id_2_token)}, {len(self.token_2_id)}")

    def make_switch_rule(self):
        self.switch = {
            tokenize.NAME: lambda tokens, tokval: tokens.append(
                self.token_2_id.get(tokval)) if tokval in self.token_2_id.keys() else tokens.append(
                self.token_2_id.get('<ID>')),
            tokenize.NUMBER: lambda tokens, tokval: tokens.append(self.token_2_id.get('<NUM>')),
            tokenize.STRING: lambda tokens, tokval: tokens.append(self.token_2_id.get('<STR>')),
            tokenize.OP: lambda tokens, tokval: tokens.append(
                self.token_2_id.get(tokval)) if tokval in self.token_2_id.keys() else tokens.append(
                self.token_2_id.get('<PUN>')),
            tokenize.ERRORTOKEN: lambda tokens, tokval: tokens.append(self.token_2_id.get('<SPE>')),
            tokenize.COMMENT: lambda tokens, tokval: tokens.append(self.token_2_id.get('<COM>'))
        }

    def from_text_to_id(self, text):
        data_generator = create_generator(text)
        tokens_iterator = tokenize.tokenize(data_generator)
        tokens = []
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                try:
                    self.switch[toknum](tokens, tokval)
                except KeyError:
                    pass
        except tokenize.TokenError:
            pass
        return tokens

    def fromIDToText(self, ids):
        tmp = ""
        for id in ids:
            if id > 7:
                tmp += self.id_2_token.get(id) + ' '
            elif id != 0:
                tmp += self.id_2_token.get(id)
            else:
                tmp += '\n'
                break
        return tmp.strip()

    def train_model(self):
        self.model.fit(self.X_train, self.y_train, batch_size=self.BATCH_SIZE,
                       epochs=self.EPOCHS, verbose=1, validation_data=(self.X_test, self.y_test))

    def save_model(self, save_path):
        self.model.save(save_path)

    @staticmethod
    def fit_on_texts(texts, vocab_path, limit=-1):
        generator = create_generator(texts)
        tokens_generator = tokenize.tokenize(generator)
        token_list = []
        try:
            for toknum, tokval, _, _, _ in tokens_generator:
                if toknum == tokenize.NAME or toknum == tokenize.OP:
                    token_list.append((toknum, tokval))
        except tokenize.TokenError as e:
            print("Tokenerror!" + str(e))

        logging.info(f"扫描的token总数{len(token_list)}")
        df_tokens = pd.DataFrame(token_list, columns=['toknum', 'tokval'])
        counter = Counter(df_tokens['tokval'])
        df_vocab = pd.DataFrame(data=counter.most_common(), columns=['token', 'count'])
        df_vocab.dropna(inplace=True)
        df_vocab.reset_index(drop=True, inplace=True)
        logging.info(f"df_vocab: {len(df_vocab)}")
        if limit != -1:
            df_vocab = df_vocab[:limit]
        logging.info(f"shrinked df_vocab: {len(df_vocab)}")
        df_vocab.to_csv(vocab_path)


def test_create_vocab():
    df_train = pd.read_pickle('../datasets/df_train_line.tar.bz2')
    logging.info(f"df_train: {len(df_train['data'])}")
    TokenModel.fit_on_texts(df_train['data'], '../vocabs/df_vocab_50000.csv', limit=50000)


def test_token_model():
    model = TokenModel('../vocabs/df_vocab_50000.csv')
    model.load_vocab()
    df_train = pd.read_pickle('../datasets/df_train_line.tar.bz2')
    model.load_datasets(df_train, limit=20000)
    model.make_model()
    model.train_model()
    model.save_model('../models/lstm_token_50000.hdf5')


def test_character_model():
    model = CharacterModel()
    model.load_datasets('../datasets/df_train_line.tar.bz2', limit=20000)
    model.make_model()
    model.train_model()
    model.save_model('../models/lstm_character_50000.hdf5')


def test_incremental_training():
    df_increment = pd.read_csv('../datasets/increment.csv')

    model = CharacterModel()
    model.load_model('../models/lstm_character_50000.hdf5')
    model.load_datasets(df_increment)
    model.train_model()
    model.save_model('../models/lstm_character_50000_increment.hdf5')

    model = TokenModel('../vocabs/nosplit_keyword_vocab50000.txt')
    model.load_model('../models/lstm_token_50000.hdf5')
    model.load_vocab()
    model.load_datasets(df_increment)
    model.train_model()
    model.save_model('../models/lstm_token_50000_increment.hdf5')


def main():
    logging.basicConfig(
        # filename='app.log',
        level=logging.DEBUG
    )
    # test_create_vocab()
    # test_token_model()
    # test_character_model()
    test_incremental_training()


if __name__ == '__main__':
    main()
