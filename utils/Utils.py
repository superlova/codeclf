# -*- coding: utf-8 -*-
# @Author  : superlova
# @Time    : 2020/11/2 15:42
# @Function:

import errno
import os
import signal
from time import process_time
from functools import wraps

import numpy as np
import six

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

from tensorflow.keras.callbacks import Callback

import logging

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def timethis(func):
    """计时函数装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = process_time()
        r = func(*args, **kwargs)
        end = process_time()
        print('{} executing time: {}s'.format(func.__name__, end - start))
        return r
    return wrapper


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


def split_at_upper_case(s):
    res = []
    j = 0
    for i in range(1, len(s)):
        if s[i - 1].islower() and s[i].isupper() \
                or s[i - 1].isalpha() and s[i].isnumeric() \
                or s[i - 1].isnumeric() and s[i].isalpha():
            res.append(s[j:i])
            j = i
    else:
        res.append(s[j:])
    return res


def split_at_under_slash(s):
    return [e for e in s.split('_') if len(e) > 0]


def split_token(s):
    res = []
    for token in split_at_under_slash(s):
        res.extend(split_at_upper_case(token))
    return res


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

kwlist = [
    'False',
    'None',
    'True',
    'and',
    'as',
    'assert',
    'async',
    'await',
    'break',
    'class',
    'continue',
    'def',
    'del',
    'elif',
    'else',
    'except',
    'finally',
    'for',
    'from',
    'global',
    'if',
    'import',
    'in',
    'is',
    'lambda',
    'nonlocal',
    'not',
    'or',
    'pass',
    'raise',
    'return',
    'try',
    'while',
    'with',
    'yield'
]

iskeyword = frozenset(kwlist).__contains__


def progress(percent, width=50):
    '''进度打印功能
       每次的输入是已经完成总任务的百分之多少
    '''
    if percent >= 100:
        percent = 100
    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # 字符串拼接的嵌套使用
    print('\r%s %d%%' % (show_str, percent), end='')


class Metrics(Callback):
    def __init__(self, valid_data, valid_size, valid_steps):
        super(Metrics, self).__init__()
        self.validation_data = valid_data
        self.valid_steps = valid_steps
        self.validation_label = []
        for _, label in valid_data.unbatch().take(valid_size):
            self.validation_label.append(label.numpy())
        self.validation_label = np.asarray(self.validation_label).reshape(-1, 1)
        logging.debug(f"{self.validation_label.shape}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.asarray(self.model.predict(self.validation_data, steps=self.valid_steps)).round()
        val_targ = self.validation_label
        logging.debug(f"{val_predict.shape}")

        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_auc = roc_auc_score(val_targ, val_predict)

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        logs['val_auc'] = _val_auc
        print(f"val_f1: {_val_f1} — val_precision: {_val_precision} — val_recall: {_val_recall} — val_auc: {_val_auc}")
        return