# -*- coding: utf-8 -*-
# @Author  : superlova
# @Time    : 2020/11/10 20:12
# @Function:

import os, sys
import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

from utils.CodeTokenizer import CodeTokenizer, CodeSplitTokenizer
from utils.CodeTokenizer import ContextCodeTokenizer, ContextCodeSplitTokenizer
from utils.Utils import timethis, Metrics
from preprocessing.DataProcessor import DataProcessor

import logging


class BasicModel(object):
    def __init__(self):
        self.history = None

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def save_model(self, save_path):
        if not self.model:
            print('Model not exist. Please training first.')
            return
        self.model.save(save_path)

    def plot_history(self, name='plot'):
        if not self.history:
            print('No training history. Please training first.')
            return
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('{}.pdf'.format(name))
        # plt.show()

    # def compute_metrics(self, model, validation_data):
    #     val_predict = (np.asarray(model.predict(validation_data[0]))).round()
    #     val_targ = validation_data[1]
    #     _val_acc = accuracy_score(val_targ, val_predict)
    #     _val_f1 = f1_score(val_targ, val_predict)
    #     _val_recall = recall_score(val_targ, val_predict)
    #     _val_precision = precision_score(val_targ, val_predict)
    #     _val_auc = roc_auc_score(val_targ, val_predict)
    #     return _val_acc, _val_f1, _val_precision, _val_recall, _val_auc


class CharModel(BasicModel):
    """训练字符模型
    """

    def __init__(self, embedding_dim=200, batch_size=32, epochs=15, hidden_dim=50):
        super().__init__()
        self.EMBEDDING_DIM = embedding_dim
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.HIDDEN_DIM = hidden_dim
        self.load_vocab()
        self.threshold = 3

    def load_vocab(self):
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890~!@#$%^&*()` ,./<>?;':\"[]{}=-+_\t\r\n|\\"
        self.char_to_int = dict((c, i + 3) for i, c in enumerate(alphabet))
        self.char_to_int['<PAD>'] = 0
        self.char_to_int['<UNK>'] = 1
        self.char_to_int['<SEP>'] = 2
        self.int_to_char = dict((i, c) for c, i in self.char_to_int.items())
        self.VOCAB_SIZE = len(self.char_to_int)

    def from_text_to_character_id(self, line):
        """将一行代码转换成字符序列，然后转换成id
        """

        def check_dict(word):
            if word in self.char_to_int.keys():
                return self.char_to_int.get(word)
            return self.char_to_int.get('<UNK>')

        char_array = np.asarray(list(line), dtype=str)
        int_array = np.asarray(list(map(check_dict, char_array)))
        return int_array

    def from_text_to_character_input(self, text, threshold=3, maxlen=70):
        """专门扫描一个文件中的文本，返回内容。
        """

        def check_dict(word):
            if word in self.char_to_int.keys():
                return self.char_to_int.get(word)
            return self.char_to_int.get('<UNK>')

        inputs = []
        for row in text:
            char_array = np.asarray(list(row), dtype=str)
            int_array = np.asarray(list(map(check_dict, char_array)))
            if len(int_array) >= threshold:
                inputs.append(int_array)
        return sequence.pad_sequences(np.asarray(inputs), padding='post', value=0, maxlen=maxlen)

    def from_text_to_character_input_and_index(self, text, threshold=3, maxlen=70):
        """专门扫描一个文件中的文本，返回行号和内容。
        """

        def check_dict(word):
            if word in self.char_to_int.keys():
                return self.char_to_int.get(word)
            return self.char_to_int.get('<UNK>')

        inputs = []
        indexes = []
        for index, row in enumerate(text):
            char_array = np.asarray(list(row), dtype=str)
            int_array = np.asarray(list(map(check_dict, char_array)))
            if len(int_array) >= threshold:
                inputs.append(int_array)
                indexes.append(index)
        return sequence.pad_sequences(np.asarray(inputs), padding='post', value=0, maxlen=maxlen), indexes

    @timethis
    def load_datasets(self, train_path, valid_path, frac=1.0):
        """训练前的数据导入
        """

        df_train = pd.read_pickle(train_path)
        df_valid = pd.read_pickle(valid_path)

        df_train = df_train.sample(frac=frac, random_state=42)
        df_valid = df_valid.sample(frac=frac, random_state=42)
        print(f'limiting training dataset: {len(df_train)}')
        print(f'limiting valid dataset: {len(df_valid)}')

        self.TRAIN_SIZE = len(df_train)
        self.VALID_SIZE = len(df_valid)
        self.INPUT_LENGTH = 70

        df_train_code = df_train[df_train['label'] == 0]  # 代码
        df_train_doc = df_train[df_train['label'] == 1]  # Doc
        df_valid_code = df_valid[df_valid['label'] == 0]  # 代码
        df_valid_doc = df_valid[df_valid['label'] == 1]  # Doc

        df_train_code = df_train_code[:len(df_train_doc)]
        df_valid_code = df_valid_code[:len(df_valid_doc)]

        X_train = []
        y_train = []
        for data in df_train_code['data']:
            int_array = np.asarray(list(map(self.from_text_to_character_id, data)))
            if len(int_array) >= self.threshold:
                X_train.append(int_array)
                y_train.append(0)
        for data in df_train_doc['data']:
            int_array = np.asarray(list(map(self.from_text_to_character_id, data)))
            if len(int_array) >= self.threshold:
                X_train.append(int_array)
                y_train.append(1)
        self.X_train = sequence.pad_sequences(np.asarray(X_train), padding='post', value=0, maxlen=self.INPUT_LENGTH)
        self.y_train = np.asarray(y_train)

        X_valid = []
        y_valid = []
        for data in df_valid_code['data']:
            int_array = np.asarray(list(map(self.from_text_to_character_id, data)))
            if len(int_array) >= self.threshold:
                X_valid.append(int_array)
                y_valid.append(0)
        for data in df_valid_doc['data']:
            int_array = np.asarray(list(map(self.from_text_to_character_id, data)))
            if len(int_array) >= self.threshold:
                X_valid.append(int_array)
                y_valid.append(1)
        self.X_valid = sequence.pad_sequences(np.asarray(X_valid), padding='post', value=0, maxlen=self.INPUT_LENGTH)
        self.y_valid = np.asarray(y_valid)

        logging.info(f"X_train: {len(X_train)}, X_valid: {len(X_valid)}")

    def construct_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.VOCAB_SIZE, input_length=self.INPUT_LENGTH,
                                      output_dim=self.EMBEDDING_DIM),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.HIDDEN_DIM, dropout=0.5)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.summary()

    @timethis
    def train_model(self, checkpoint_save_path, patience=5):
        def get_checkpoint_callback(checkpoint_path):
            cp_callbacks = tf.keras.callbacks.ModelCheckpoint(
                monitor='val_accuracy',
                mode='max',
                verbose=2,
                filepath=checkpoint_path,
                save_weights_only=True,
                save_best_only=True)
            return cp_callbacks

        def get_earlystop_callback(patience=5):
            es_callbacks = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=patience
            )
            return es_callbacks

        if os.path.exists(checkpoint_save_path + '.index'):
            print('Checkpoint detected, loading the model...')
            self.model.load_weights(checkpoint_save_path)
        check_point = get_checkpoint_callback(checkpoint_save_path)
        early_stopping = get_earlystop_callback(patience)

        self.history = self.model.fit(self.X_train, self.y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                                      validation_data=(self.X_valid, self.y_valid),
                                      callbacks=[early_stopping, check_point])

    @timethis
    def evaluate(self, test_path):
        if not self.model:
            print('Model not exist. Please training first.')
            return
        if not self.X_test:
            df_test = pd.read_pickle(test_path)
            self.TEST_SIZE = len(df_test)
            X_test = []
            y_test = []
            for data, label in zip(df_test['data'], df_test['label']):
                int_array = np.asarray(list(map(self.from_text_to_character_id, data)))
                if len(int_array) >= self.threshold:
                    X_test.append(int_array)
                    y_test.append(label)
            self.X_test = sequence.pad_sequences(np.asarray(X_test), padding='post', value=0, maxlen=self.INPUT_LENGTH)
            self.y_test = np.asarray(y_test)

            logging.info(f"X_test: {len(X_test)}")

        loss, acc = self.model.evaluate(X=self.X_test, y=self.y_test, batch_size=self.BATCH_SIZE)
        print(f'loss: {loss}, acc: {acc}')


class ClfModel(BasicModel):
    def __init__(self, embedding_dim=200, batch_size=32, epochs=15, hidden_dim=50):
        super().__init__()
        self.EMBEDDING_DIM = embedding_dim
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.HIDDEN_DIM = hidden_dim

    def load_vocab(self, vocab_path):
        self.tokenizer = CodeTokenizer(vocab_path)
        self.VOCAB_SIZE = len(self.tokenizer.vocab)

    @timethis
    def load_datasets(self, train_path, valid_path, frac=1.0):
        df_train = pd.read_pickle(train_path)
        df_valid = pd.read_pickle(valid_path)

        # if limit_dataset != -1:
        #     print(f'limiting training dataset: {limit_dataset}')
        df_train = df_train.sample(frac=frac, random_state=42)
        df_valid = df_valid.sample(frac=frac, random_state=42)
        print(f'limiting training dataset: {len(df_train)}')
        print(f'limiting valid dataset: {len(df_valid)}')

        self.TRAIN_SIZE = len(df_train)
        self.VALID_SIZE = len(df_valid)
        self.INPUT_LENGTH = 30

        def make_ds(lines, labels):
            ds = tf.data.Dataset.from_tensor_slices((lines, labels))
            ds = ds.shuffle(1000000).repeat()
            return ds

        train_code = self.tokenizer.from_lines_to_token_input(df_train[df_train['label'] == 0]['data'])
        train_docs = self.tokenizer.from_lines_to_token_input(df_train[df_train['label'] == 1]['data'])
        train_code_ds = make_ds(train_code, np.zeros((len(train_code))))
        train_docs_ds = make_ds(train_docs, np.ones((len(train_docs))))
        self.train_ds = tf.data.experimental.sample_from_datasets([train_code_ds, train_docs_ds], weights=[0.5, 0.5])
        self.train_ds = self.train_ds.batch(self.BATCH_SIZE).prefetch(2)

        val_data, val_label = self.tokenizer.from_df_to_token_input(df_valid)
        logging.info(f'len(val_data): {len(val_data)}, {len(val_label)}')
        self.val_ds = make_ds(val_data, val_label)
        self.val_ds = self.val_ds.batch(self.BATCH_SIZE).prefetch(2)

    def construct_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.VOCAB_SIZE, input_length=self.INPUT_LENGTH,
                                      output_dim=self.EMBEDDING_DIM),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.HIDDEN_DIM, return_sequences=True, dropout=0.5)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.HIDDEN_DIM, return_sequences=True, dropout=0.5)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.HIDDEN_DIM, dropout=0.5)),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.summary()

    @timethis
    def train_model(self, checkpoint_save_path, patience=3):
        def get_checkpoint_callback(checkpoint_path):
            cp_callbacks = tf.keras.callbacks.ModelCheckpoint(
                monitor='val_f1',
                mode='max',
                verbose=2,
                filepath=checkpoint_path,
                save_weights_only=True,
                save_best_only=True)
            return cp_callbacks

        def get_earlystop_callback(patience=3):
            es_callbacks = tf.keras.callbacks.EarlyStopping(
                monitor='val_f1', patience=patience
            )
            return es_callbacks

        if os.path.exists(checkpoint_save_path + '.index'):
            print('Checkpoint detected, loading the model...')
            self.model.load_weights(checkpoint_save_path)
        check_point = get_checkpoint_callback(checkpoint_save_path)
        early_stopping = get_earlystop_callback(patience)

        steps_per_epoch = tf.math.ceil(self.TRAIN_SIZE / self.BATCH_SIZE).numpy()
        validation_steps = tf.math.ceil(self.VALID_SIZE / self.BATCH_SIZE).numpy()

        self.history = self.model.fit(self.train_ds, epochs=self.EPOCHS, validation_data=self.val_ds,
                                      callbacks=[early_stopping, check_point],
                                      steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    @timethis
    def evaluate(self, test_path):
        if not self.model:
            print('Model not exist. Please training first.')
            return
        if not self.test_ds:
            df_test = pd.read_pickle(test_path)
            self.TEST_SIZE = len(df_test)

            def make_ds(lines, labels):
                ds = tf.data.Dataset.from_tensor_slices((lines, labels))
                ds = ds.shuffle(1000000).repeat()
                return ds

            test_data, test_label = self.tokenizer.from_df_to_token_input(df_test)
            self.test_ds = make_ds(test_data, test_label)
            self.test_ds = self.test_ds.batch(self.BATCH_SIZE).prefetch(2)

        test_steps = tf.math.ceil(self.TEST_SIZE / self.BATCH_SIZE).numpy()
        loss, acc = self.model.evaluate(self.test_ds, steps=test_steps)
        print(f'loss: {loss}, acc: {acc}')


class ClfSplitModel(ClfModel):
    def __init__(self):
        super(ClfSplitModel, self).__init__()

    def load_vocab(self, vocab_path):
        self.tokenizer = CodeSplitTokenizer(vocab_path)
        self.VOCAB_SIZE = len(self.tokenizer.vocab)


class ContextModel(BasicModel):
    def __init__(self, before=1, after=1, embedding_dim=200, batch_size=1024, epochs=40, hidden_dim=50,
                 context_mode='bta'):
        super().__init__()
        self.EMBEDDING_DIM = embedding_dim
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.CONTEXT_BEFORE = before
        self.CONTEXT_AFTER = after
        self.INPUT_LENGTH = 30 * (self.CONTEXT_BEFORE + self.CONTEXT_AFTER + 1)
        self.HIDDEN_DIM = hidden_dim
        self.CONTEXT_MODE = context_mode

    def load_vocab(self, vocab_path):
        self.tokenizer = ContextCodeTokenizer(vocab_path)
        self.VOCAB_SIZE = len(self.tokenizer.vocab)

    @staticmethod
    def _feature_to_id_bta(features, label, tokenizer, input_length):
        """Context特有的方法，将dataset中的上下文结合到一起并生成ids"""
        return tokenizer.from_feature_to_token_id_bta(features[0].decode("utf-8"),
                                                      features[1].decode("utf-8"),
                                                      features[2].decode("utf-8"), maxlen=input_length), label

    @staticmethod
    def _feature_to_id_bat(features, label, tokenizer, input_length):
        """Context特有的方法，将dataset中的上下文结合到一起并生成ids"""
        return tokenizer.from_feature_to_token_id_bat(features[0].decode("utf-8"),
                                                      features[1].decode("utf-8"),
                                                      features[2].decode("utf-8"), maxlen=input_length), label

    @staticmethod
    def _feature_to_id_tba(features, label, tokenizer, input_length):
        """Context特有的方法，将dataset中的上下文结合到一起并生成ids"""
        return tokenizer.from_feature_to_token_id_tba(features[0].decode("utf-8"),
                                                      features[1].decode("utf-8"),
                                                      features[2].decode("utf-8"), maxlen=input_length), label

    @timethis
    def load_datasets(self, train_path, valid_path, frac=1.0):
        """输入corpus_path
        """
        df_train = pd.read_pickle(train_path)
        df_valid = pd.read_pickle(valid_path)
        df_train = df_train['code']
        df_valid = df_valid['code']

        # if limit_dataset != -1:
        # print(f'limiting training dataset: {limit_dataset}')
        # df_train = df_train.sample(limit_dataset, random_state=42)
        # df_valid = df_valid.sample(limit_dataset, random_state=42)
        df_train = df_train.sample(frac=frac, random_state=42)
        df_valid = df_valid.sample(frac=frac, random_state=42)
        print(f'limiting training dataset: {len(df_train)}')
        print(f'limiting valid dataset: {len(df_valid)}')

        self.TRAIN_SIZE = len(df_train)
        self.VALID_SIZE = len(df_valid)

        logging.info(f"{len(df_train)}, {len(df_valid)}")

        if self.CONTEXT_MODE == 'bta':
            feature_to_id = functools.partial(self._feature_to_id_bta, tokenizer=self.tokenizer,
                                              input_length=self.INPUT_LENGTH)
        elif self.CONTEXT_MODE == 'bat':
            feature_to_id = functools.partial(self._feature_to_id_bat, tokenizer=self.tokenizer,
                                              input_length=self.INPUT_LENGTH)
        elif self.CONTEXT_MODE == 'tba':
            feature_to_id = functools.partial(self._feature_to_id_tba, tokenizer=self.tokenizer,
                                              input_length=self.INPUT_LENGTH)
        else:
            feature_to_id = functools.partial(self._feature_to_id_bta, tokenizer=self.tokenizer,
                                              input_length=self.INPUT_LENGTH)

        def tf_feature_to_id(features, label):
            """Context特有的方法，将feature_to_id_bta包装为tf.py_function"""
            label_shape = label.shape
            [features, label] = tf.numpy_function(feature_to_id,
                                                  inp=[features, label],
                                                  Tout=[tf.int32, tf.int64])
            features.set_shape((self.INPUT_LENGTH,))
            label.set_shape(label_shape)
            return features, label

        dp = DataProcessor()
        ds_train = dp.process_context_tfdata_merge(df_train, self.CONTEXT_BEFORE, self.CONTEXT_AFTER)
        ds_valid = dp.process_context_tfdata_merge(df_valid, self.CONTEXT_BEFORE, self.CONTEXT_AFTER, reshuffle=False)

        ds_train = ds_train.map(tf_feature_to_id, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        train_code_ds = (ds_train
                         .filter(lambda features, label: label == 0)
                         .shuffle(100000)
                         .repeat())
        train_docs_ds = (ds_train
                         .filter(lambda features, label: label == 1)
                         .shuffle(100000)
                         .repeat())
        self.train_ds = tf.data.experimental.sample_from_datasets(
            [train_code_ds, train_docs_ds],
            weights=[0.5, 0.5])
        self.train_ds = self.train_ds.batch(self.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

        self.val_ds = ds_valid.map(tf_feature_to_id, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.val_ds = self.val_ds.batch(self.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    def _make_lstm_1(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.VOCAB_SIZE, input_length=self.INPUT_LENGTH,
                                      output_dim=self.EMBEDDING_DIM),
            tf.keras.layers.LSTM(self.HIDDEN_DIM, dropout=0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def _make_lstm_3(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.VOCAB_SIZE, input_length=self.INPUT_LENGTH,
                                      output_dim=self.EMBEDDING_DIM),
            tf.keras.layers.LSTM(self.HIDDEN_DIM, return_sequences=True, dropout=0.5),
            tf.keras.layers.LSTM(self.HIDDEN_DIM, return_sequences=True, dropout=0.5),
            tf.keras.layers.LSTM(self.HIDDEN_DIM, dropout=0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def _make_bilstm_1(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.VOCAB_SIZE, input_length=self.INPUT_LENGTH,
                                      output_dim=self.EMBEDDING_DIM),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.HIDDEN_DIM, dropout=0.5)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def _make_bilstm_3(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.VOCAB_SIZE, input_length=self.INPUT_LENGTH,
                                      output_dim=self.EMBEDDING_DIM),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.HIDDEN_DIM, return_sequences=True, dropout=0.5)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.HIDDEN_DIM, return_sequences=True, dropout=0.5)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.HIDDEN_DIM, dropout=0.5)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def _make_bilstm_3_dense(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.VOCAB_SIZE, input_length=self.INPUT_LENGTH,
                                      output_dim=self.EMBEDDING_DIM),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.HIDDEN_DIM, return_sequences=True, dropout=0.5)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.HIDDEN_DIM, return_sequences=True, dropout=0.5)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.HIDDEN_DIM, dropout=0.5)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def construct_model(self, model_type='lstm_1'):
        if model_type == 'lstm_1':
            self.model = self._make_lstm_1()
        elif model_type == 'lstm_3':
            self.model = self._make_lstm_3()
        elif model_type == 'bilstm_1':
            self.model = self._make_bilstm_1()
        elif model_type == 'bilstm_3':
            self.model = self._make_bilstm_3()
        elif model_type == 'bilstm_3_dense':
            self.model = self._make_bilstm_3_dense()
        else:
            self.model = self._make_bilstm_3_dense()

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.summary()

    @timethis
    def train_model(self, checkpoint_save_path, patience=5):
        def get_checkpoint_callback(checkpoint_path):
            cp_callbacks = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                save_best_only=True)
            return cp_callbacks

        def get_earlystop_callback(patience=5):
            es_callbacks = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience
            )
            return es_callbacks

        if os.path.exists(checkpoint_save_path + '.index'):
            print('Checkpoint detected, loading the model...')
            self.model.load_weights(checkpoint_save_path)
        check_point = get_checkpoint_callback(checkpoint_save_path)
        early_stopping = get_earlystop_callback(patience)

        steps_per_epoch = tf.math.ceil(self.TRAIN_SIZE / self.BATCH_SIZE).numpy()
        validation_steps = tf.math.ceil(self.VALID_SIZE / self.BATCH_SIZE).numpy()

        logging.info("metrics start")
        metrics = Metrics(valid_data=self.val_ds,
                          valid_steps=validation_steps)
        logging.info("metric end")

        self.history = self.model.fit(self.train_ds, epochs=self.EPOCHS, validation_data=self.val_ds,
                                      callbacks=[metrics,
                                                 check_point,
                                                 early_stopping],
                                      steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    @timethis
    def evaluate(self, test_path):
        """corpus_path"""
        if not self.model:
            print('Model not exist. Please training first.')
            return
        # if not self.test_ds:
        df_test = pd.read_pickle(test_path)
        df_test = df_test['code']
        self.TEST_SIZE = len(df_test)

        if self.CONTEXT_MODE == 'bta':
            feature_to_id = functools.partial(self._feature_to_id_bta, tokenizer=self.tokenizer,
                                              input_length=self.INPUT_LENGTH)
        elif self.CONTEXT_MODE == 'bat':
            feature_to_id = functools.partial(self._feature_to_id_bat, tokenizer=self.tokenizer,
                                              input_length=self.INPUT_LENGTH)
        elif self.CONTEXT_MODE == 'tba':
            feature_to_id = functools.partial(self._feature_to_id_tba, tokenizer=self.tokenizer,
                                              input_length=self.INPUT_LENGTH)
        else:
            feature_to_id = functools.partial(self._feature_to_id_bta, tokenizer=self.tokenizer,
                                              input_length=self.INPUT_LENGTH)

        def tf_feature_to_id(features, label):
            """Context特有的方法，将feature_to_id_bta包装为tf.py_function"""
            label_shape = label.shape
            [features, label] = tf.numpy_function(feature_to_id,
                                                  inp=[features, label],
                                                  Tout=[tf.int32, tf.int64])
            features.set_shape((self.INPUT_LENGTH,))
            label.set_shape(label_shape)
            return features, label

        dp = DataProcessor()
        ds_test = dp.process_context_tfdata_merge(df_test, self.CONTEXT_BEFORE, self.CONTEXT_AFTER)
        self.test_ds = ds_test.map(tf_feature_to_id, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.test_ds = self.test_ds.batch(self.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

        test_steps = tf.math.ceil(self.TEST_SIZE / self.BATCH_SIZE).numpy()
        loss, acc = self.model.evaluate(self.test_ds, steps=test_steps)
        print(f'loss: {loss}, acc: {acc}')


class ContextSpiltModel(ContextModel):
    def __init__(self, before=1, after=1, embedding_dim=200, batch_size=1024, epochs=40, hidden_dim=50,
                 context_mode='bta'):
        super(ContextSpiltModel, self).__init__(before, after, embedding_dim, batch_size, epochs, hidden_dim,
                                                context_mode)

    def load_vocab(self, vocab_path):
        self.tokenizer = ContextCodeSplitTokenizer(vocab_path)
        self.VOCAB_SIZE = len(self.tokenizer.vocab)


#########################################

def test_clf_split_model():
    logging.basicConfig(
        level=logging.INFO
    )

    trainer = ClfSplitModel()
    trainer.load_vocab('../vocabs/split_keyword_vocab50000.txt')

    logging.info('loading training data...')
    trainer.load_datasets(train_path='../datasets/df_train_line.tar.bz2',
                          valid_path='../datasets/df_valid_line.tar.bz2',
                          limit_dataset=20000)

    logging.info('datasets ready!')
    trainer.construct_model()

    logging.info('start training...')
    trainer.train_model(checkpoint_save_path='../checkpoint/lstm_model_token_50000_bi')

    logging.info('saving model...')
    trainer.save_model('../models/lstm_model_token_50000_bi.hdf5')

    trainer.plot_history()

    logging.info('evaluating model...')
    trainer.evaluate(test_path='../datasets/df_test_line.tar.bz2')


def test_context_tokenizer():
    corpus_path = '../datasets/df_test_corpus.tar.bz2'
    df_data = pd.read_pickle(corpus_path)
    data = df_data['code']

    ccst = ContextCodeSplitTokenizer(os.path.join(project_dir, 'vocabs/split_simple_vocab50000.txt'))
    dp = DataProcessor()

    ds = dp.process_context_tfdata_merge(data)
    for features, label in ds.take(10):
        # print(features, len(features), label)
        print(ccst.from_feature_to_token_id_bta(features[0].numpy().decode("utf-8"),
                                                features[1].numpy().decode("utf-8"),
                                                features[2].numpy().decode("utf-8")))


def test_count_token_avg():
    corpus_path = '../datasets/df_test_corpus.tar.bz2'
    df_data = pd.read_pickle(corpus_path)
    data = df_data['code']

    ccst = ContextCodeSplitTokenizer(os.path.join(project_dir, 'vocabs/split_simple_vocab50000.txt'))
    dp = DataProcessor()

    ds = dp.process_context_tfdata_merge(data)

    token_lengths = []
    for features, label in ds:
        # print(features, len(features), label)
        for f in features:
            tokens, _ = ccst.tokenize_with_type(f.numpy().decode("utf-8"))
            token_lengths.append(len(tokens))

    import matplotlib.pyplot as plt
    import seaborn as sns
    # fig, ax = plt.subplots()
    ax = sns.distplot(token_lengths)
    plt.show()


def test_tokenize_map():
    def feature_to_id_bta(features, label):
        return ccst.from_feature_to_token_id_bta(features[0].numpy().decode("utf-8"),
                                                 features[1].numpy().decode("utf-8"),
                                                 features[2].numpy().decode("utf-8")), label

    corpus_path = '../datasets/df_test_corpus.tar.bz2'
    df_data = pd.read_pickle(corpus_path)
    data = df_data['code']

    ccst = ContextCodeSplitTokenizer(os.path.join(project_dir, 'vocabs/split_simple_vocab50000.txt'))
    dp = DataProcessor()

    ds = dp.process_context_tfdata_merge(data)
    # ds_features = (features[0] for features in ds)
    # print(ds_features)

    ds = ds.map(lambda features, label:
                tf.py_function(feature_to_id_bta,
                               inp=[features, label],
                               Tout=[tf.string, tf.int32]))

    for feature, label in ds:
        print(feature)
        print(label)


def test_context_model():
    trainer = ContextModel(before=1, after=1, context_mode='bta')
    trainer.load_vocab('../vocabs/nosplit_keyword_vocab50000.txt')

    logging.info('loading training data...')
    trainer.load_datasets(train_path='../datasets/df_train_corpus.tar.bz2',
                          valid_path='../datasets/df_valid_corpus.tar.bz2',
                          frac=0.2)

    logging.info('datasets ready!')
    trainer.construct_model(model_type='bilstm_3_dense')

    logging.info('start training...')
    trainer.train_model(checkpoint_save_path='../checkpoint/bilstm_3_dense')

    logging.info('saving model...')
    trainer.save_model('../models/bilstm_3_dense.hdf5')

    trainer.plot_history()

    logging.info('evaluating model...')
    trainer.evaluate(test_path='../datasets/df_test_corpus.tar.bz2')


def test_context_split_model():
    trainer = ContextSpiltModel(before=0, after=0, context_mode='bat')
    trainer.load_vocab('../vocabs/split_keyword_vocab50000.txt')

    logging.info('loading training data...')
    trainer.load_datasets(train_path='../datasets/df_train_corpus.tar.bz2',
                          valid_path='../datasets/df_valid_corpus.tar.bz2',
                          frac=0.1)

    logging.info('datasets ready!')
    trainer.construct_model(model_type='lstm_1')

    logging.info('start training...')
    trainer.train_model(checkpoint_save_path='../checkpoint/lstm_1_bat00data50000vocab50000split_1')

    logging.info('saving model...')
    trainer.save_model('../models/lstm_1_bat00data50000vocab50000split_1.hdf5')

    trainer.plot_history()

    logging.info('evaluating model...')
    trainer.evaluate(test_path='../datasets/df_test_corpus.tar.bz2')


def test_valid_data():
    trainer = ContextSpiltModel(before=0, after=0, context_mode='bat')
    trainer.load_vocab('../vocabs/split_keyword_vocab50000.txt')

    # logging.info('loading training data...')
    trainer.load_datasets(train_path='../datasets/df_train_corpus.tar.bz2',
                          valid_path='../datasets/df_valid_corpus.tar.bz2',
                          frac=0.1)
    print("trainer.VALID_SIZE: ", trainer.VALID_SIZE)
    print("trainer.BATCH_SIZE: ", trainer.BATCH_SIZE)

    validation_label = []
    for _, batch_label in trainer.val_ds.take(3):
        for label in batch_label:
            validation_label.append(label.numpy())
    print("len(validation_label): ", len(validation_label))
    print("validation_label: ", validation_label)
    print("sum of label", np.sum(validation_label) / len(validation_label))

    trainer.construct_model(model_type='lstm_1')
    validation_steps = tf.math.ceil(trainer.VALID_SIZE / trainer.BATCH_SIZE).numpy()
    print("validation_steps: ", validation_steps)

    # logging.info("metrics start")
    # metrics = Metrics(valid_data=trainer.val_ds,
    #                   valid_size=trainer.VALID_SIZE,
    #                   valid_steps=validation_steps)
    # logging.info("metric end")
    #
    # self.history = self.model.fit(self.train_ds, epochs=self.EPOCHS, validation_data=self.val_ds,
    #                               callbacks=[metrics,
    #                                          check_point,
    #                                          early_stopping],
    #                               steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    val_predict = np.asarray(trainer.model.predict(trainer.val_ds, steps=validation_steps)).round()
    val_predict = np.squeeze(val_predict)
    print("len(validation_label): ", len(val_predict))
    print("validation_label: ", val_predict)


def test_char_model():
    cm = CharModel()
    cm.load_datasets(train_path='../datasets/df_train_line.tar.bz2',
                     valid_path='../datasets/df_valid_line.tar.bz2',
                     frac=0.05)
    cm.construct_model()
    cm.train_model(checkpoint_save_path='../checkpoint/char_model_1')
    logging.info('saving model...')
    cm.save_model('../models/char_model_1.hdf5')

    cm.plot_history()

    logging.info('evaluating model...')
    cm.evaluate(test_path='../datasets/df_test_line.tar.bz2')


def main():
    logging.basicConfig(
        level=logging.INFO
    )
    # test_clf_split_model()
    # test_context_tokenizer()
    # test_count_token_avg()
    # test_tokenize_map()
    # test_context_model()
    # test_context_split_model()
    # test_valid_data()
    test_char_model()


if __name__ == '__main__':
    main()
