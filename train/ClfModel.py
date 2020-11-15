# -*- coding: utf-8 -*-
# @Author  : superlova
# @Time    : 2020/11/10 20:12
# @Function:

import os

import numpy as np
import pandas as pd

import tensorflow as tf

from utils.CodeTokenizer import CodeTokenizer, CodeSplitTokenizer
from utils.Utils import timethis

import logging


class ClfModel(object):
    def __init__(self):
        self.EMBEDDING_DIM = 200
        self.BATCH_SIZE = 32
        self.EPOCHS = 15

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def load_vocab(self, vocab_path):
        self.tokenizer = CodeTokenizer(vocab_path)
        self.VOCAB_SIZE = len(self.tokenizer.vocab)

    @timethis
    def load_datasets(self, train_path, valid_path, limit_dataset=-1):
        df_train = pd.read_pickle(train_path)
        df_valid = pd.read_pickle(valid_path)

        if limit_dataset != -1:
            print(f'limiting training dataset: {limit_dataset}')
            df_train = df_train.sample(limit_dataset, random_state=42)
            df_valid = df_valid.sample(limit_dataset, random_state=42)

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
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.5)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.5)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, dropout=0.5)),
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
                filepath=checkpoint_path,
                save_weights_only=True,
                save_best_only=True)
            return cp_callbacks

        def get_earlystop_callback(patience=3):
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

        self.history = self.model.fit(self.train_ds, epochs=self.EPOCHS, validation_data=self.val_ds,
                                      callbacks=[early_stopping, check_point],
                                      steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    def plot_history(self):
        if not self.history:
            print('No training history. Please training first.')
            return
        import matplotlib.pyplot as plt
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    @timethis
    def save_model(self, save_path):
        if not self.model:
            print('Model not exist. Please training first.')
            return
        self.model.save(save_path)

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


def main():
    test_clf_split_model()


if __name__ == '__main__':
    main()
