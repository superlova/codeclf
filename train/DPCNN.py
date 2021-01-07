'''
Date: 2021-01-07 23:13:31
LastEditors: superlova
LastEditTime: 2021-01-07 23:13:31
FilePath: /codeclf/train/DPCNN.py
'''

import os
import abc
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, Conv1D, Flatten, Dense, Activation, Add, MaxPooling1D
from sklearn.metrics import accuracy_score

from train.BASE_MODEL import BaseModel


class KerasBaseModel(BaseModel):
    def __init__(self, config, **kwargs):
        super(KerasBaseModel, self).__init__()
        self.config = config
        self.level = self.config.input_level
        self.max_len = self.config.max_len[self.config.input_level]
        self.word_embeddings = config.word_embeddings
        self.n_class = config.n_class

        self.callbacks = []
        self.init_callbacks()

        self.model = self.build(**kwargs)

    def init_callbacks(self):
        self.callbacks.append(ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name),
            monitor=self.config.checkpoint_monitor,
            save_best_only=self.config.checkpoint_save_best_only,
            save_weights_only=self.config.checkpoint_save_weights_only,
            mode=self.config.checkpoint_save_weights_mode,
            verbose=self.config.checkpoint_verbose
        ))

        self.callbacks.append(EarlyStopping(
            monitor=self.config.early_stopping_monitor,
            mode=self.config.early_stopping_mode,
            patience=self.config.early_stopping_patience,
            verbose=self.config.early_stopping_verbose
        ))

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def load_best_model(self):
        logging.info('loading model checkpoint: %s.hdf5\n' % self.config.exp_name)
        self.load_weights(os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name))
        logging.info('Model loaded')

    @abc.abstractmethod
    def build(self):
        """Build the model"""

    def train(self, data_train, data_dev=None):
        x_train, y_train = data_train

        logging.info('start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size, epochs=self.config.n_epoch,
                       validation_split=0.1, validation_data=data_dev, callbacks=self.callbacks)
        logging.info('training end...')

    def evaluate(self, data):
        input_data, label = data
        prediction = self.predict(input_data)
        acc = accuracy_score(label, prediction)
        logging.info('acc : %f', acc)
        return acc

    def predict(self, data):
        return self.model.predict(data)


class DPCNN(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(DPCNN, self).__init__(config, **kwargs)

    def build(self):
        input_text = Input(shape=(self.max_len,))

        embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                    weights=[self.word_embeddings],
                                    trainable=self.config.word_embed_trainable)(input_text)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)

        repeat = 3
        size = self.max_len
        region_x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(text_embed)
        x = Activation(activation='relu')(region_x)
        x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
        x = Activation(activation='relu')(x)
        x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
        x = Add()([x, region_x])

        for _ in range(repeat):
            px = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
            size = int((size + 1) / 2)
            x = Activation(activation='relu')(px)
            x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
            x = Activation(activation='relu')(x)
            x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
            x = Add()([x, px])

        x = MaxPooling1D(pool_size=size)(x)
        sentence_embed = Flatten()(x)

        dense_layer = Dense(256, activation='relu')(sentence_embed)
        output = Dense(self.n_class, activation='softmax')(dense_layer)

        model = Model(input_text, output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model