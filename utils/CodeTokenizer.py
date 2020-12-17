# -*- coding: utf-8 -*-
# @Author  : superlova
# @Time    : 2020/11/9 9:05
# @Function:

'''
Author: Zyt
Date: 2020-11-09 09:05:21
LastEditTime: 2020-12-17 23:57:08
LastEditors: superlova
Description: 将文本转化为id，包含许多不同任务的tokenizer
FilePath: \codeclf\utils\CodeTokenizer.py
'''

import tokenize

import numpy as np
import pandas as pd

import sys
project_dir = 'C:/Users/zyt/Documents/GitHub Repositories/codeclf_gui/codeclf'
sys.path.append(project_dir)

from utils.Utils import create_generator
from utils.Utils import pad_sequences
from utils.Utils import split_token
from utils.Utils import iskeyword

import logging


class BaseCodeTokenizer(object):
    def __init__(self, vocab_file):
        self.init_vocab(vocab_file)

    def init_vocab(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)
        self.vocab = {v: k + 7 for k, v in enumerate(self.vocab)}
        self.vocab['<PAD>'] = 0  # padding position
        self.vocab['<UID>'] = 1  # unknown_id, toknum==1
        self.vocab['<PUN>'] = 2  # punctuation, toknum==54
        self.vocab['<NUM>'] = 3  # number, toknum==2
        self.vocab['<STR>'] = 4  # string, toknum==3
        self.vocab['<SPE>'] = 5  # special character, toknum==59
        self.vocab['<COM>'] = 6  # comment, toknum==60
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @staticmethod
    def load_vocab(vocab_file):
        vocab = []
        with open(vocab_file, 'r', encoding='utf8') as f:
            for line in f:
                vocab.append(line.rstrip('\n'))
        return vocab

    def tokenize(self, row):
        tokens, _ = self.tokenize_with_type(row)
        return tokens

    def tokenize_with_type(self, row):
        """检查有没有相邻的两个id, 有则返回true"""
        data_generator = create_generator([row])
        tokens_iterator = tokenize.tokenize(data_generator)
        tokens = []
        tokens_type = []
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                tokens.append(tokval)
                tokens_type.append(toknum)
        except tokenize.TokenError:
            pass
        return tokens, tokens_type


class BERTCodeTokenizer(BaseCodeTokenizer):
    """TODO
    """
    def __init__(self, vocab_file):
        # super().__init__(vocab_file)
        self.init_vocab(vocab_file)

    def init_vocab(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)
        self.vocab = {v: k + 10 for k, v in enumerate(self.vocab)}
        self.vocab['<PAD>'] = 0  # padding position
        self.vocab['<CLS>'] = 1  # added in front of every input sample (BERT)
        self.vocab['<SEP>'] = 2  # separator token (BERT)
        self.vocab['<MSK>'] = 3  # mask token (BERT)
        self.vocab['<UID>'] = 4  # unknown_id, toknum==1
        self.vocab['<PUN>'] = 5  # punctuation, toknum==54
        self.vocab['<NUM>'] = 6  # number, toknum==2
        self.vocab['<STR>'] = 7  # string, toknum==3
        self.vocab['<SPE>'] = 8  # special character, toknum==59
        self.vocab['<COM>'] = 9 # comment, toknum==60
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def from_row_to_token_id(self, row):
        """把一行代码转成token"""
        raise NotImplementedError

    def from_lines_to_token_input(self, lines, threshold=3, maxlen=30):
        """输入[line]，输出适合直接学习的[input]与对应的[index]。
        threshold目的是筛选那些长度过短的line
        maxlen是对齐[input]长度，方便模型输入
        加index是为了能够溯源，防止index被打乱"""
        raise NotImplementedError

    def from_df_to_token_input(self, df, threshold=3, maxlen=30):
        raise NotImplementedError

    def from_lines_to_token_input_yield(self, lines, threshold=3, maxlen=30):
        """输入[line]，输出适合直接学习的[input]与对应的[index]。
        threshold目的是筛选那些长度过短的line
        maxlen是对齐[input]长度，方便模型输入
        加index是为了能够溯源，防止index被打乱"""
        raise NotImplementedError

    def from_lines_to_token_input_and_index(self, lines, threshold=3, maxlen=30):
        """输入[line]，输出适合直接学习的[input]与对应的[index]。
        threshold目的是筛选那些长度过短的line
        maxlen是对齐[input]长度，方便模型输入
        加index是为了能够溯源，防止index被打乱"""
        raise NotImplementedError

    @staticmethod
    def is_adjacent_id(row):
        """检查有没有相邻的两个id, 有则返回true"""
        raise NotImplementedError


class CodeTokenizer(BaseCodeTokenizer):
    def __init__(self, vocab_file):
        super().__init__(vocab_file)

    def from_row_to_token_id(self, row):
        """把一行代码转成token"""
        data_generator = create_generator([row])
        tokens_iterator = tokenize.tokenize(data_generator)
        ids = []
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                if toknum == tokenize.NAME:
                    ids.append(
                        self.vocab.get(tokval)) if tokval in self.vocab.keys() else ids.append(
                        self.vocab.get('<UID>'))
                elif toknum == tokenize.NUMBER:
                    ids.append(self.vocab.get('<NUM>'))
                elif toknum == tokenize.STRING:
                    ids.append(self.vocab.get('<STR>'))
                elif toknum == tokenize.OP:
                    ids.append(
                        self.vocab.get(tokval)) if tokval in self.vocab.keys() else ids.append(
                        self.vocab.get('<PUN>'))
                elif toknum == tokenize.ERRORTOKEN:
                    ids.append(self.vocab.get('<SPE>'))
                elif toknum == tokenize.COMMENT:
                    ids.append(self.vocab.get('<COM>'))
        except tokenize.TokenError:
            pass  # 遍历到末尾会raise error
        return ids

    def from_lines_to_token_input(self, lines, threshold=3, maxlen=30):
        """输入[line]，输出适合直接学习的[input]与对应的[index]。
        threshold目的是筛选那些长度过短的line
        maxlen是对齐[input]长度，方便模型输入
        加index是为了能够溯源，防止index被打乱"""
        inputs = []
        for row in lines:
            int_array = np.asarray(self.from_row_to_token_id(row))
            if len(int_array) >= threshold:
                inputs.append(int_array)
            else:
                logging.debug(row)
        return pad_sequences(np.asarray(inputs), padding='post', value=0, maxlen=maxlen)

    def from_df_to_token_input(self, df, threshold=3, maxlen=30):
        inputs = []
        labels = []
        for index, row in df.iterrows():
            int_array = np.asarray(self.from_row_to_token_id(row[0]))
            if len(int_array) >= threshold:
                inputs.append(int_array)
                labels.append(row[1])
            else:
                logging.debug(row[0] + '\t' + str(row[1]))
        return pad_sequences(np.asarray(inputs), padding='post', value=0, maxlen=maxlen), labels

    def from_lines_to_token_input_yield(self, lines, threshold=3, maxlen=30):
        """输入[line]，输出适合直接学习的[input]与对应的[index]。
        threshold目的是筛选那些长度过短的line
        maxlen是对齐[input]长度，方便模型输入
        加index是为了能够溯源，防止index被打乱"""
        for row in lines:
            int_array = np.asarray(self.from_row_to_token_id(row))
            if len(int_array) >= threshold:
                yield pad_sequences(np.asarray([int_array]), padding='post', value=0, maxlen=maxlen)[0]
        # return pad_sequences(np.asarray(inputs), padding='post', value=0, maxlen=maxlen)

    def from_lines_to_token_input_and_index(self, lines, threshold=3, maxlen=30):
        """输入[line]，输出适合直接学习的[input]与对应的[index]。
        threshold目的是筛选那些长度过短的line
        maxlen是对齐[input]长度，方便模型输入
        加index是为了能够溯源，防止index被打乱"""
        inputs = []
        indexes = []
        for index, row in enumerate(lines):
            int_array = np.asarray(self.from_row_to_token_id(row))
            if len(int_array) >= threshold:
                indexes.append(index)
                inputs.append(int_array)
        return pad_sequences(np.asarray(inputs), padding='post', value=0, maxlen=maxlen), indexes

    @staticmethod
    def is_adjacent_id(row):
        """检查有没有相邻的两个id, 有则返回true"""
        data_generator = create_generator([row])
        tokens_iterator = tokenize.tokenize(data_generator)
        res = []
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                res.append((toknum, tokval))
        except tokenize.TokenError:
            pass
        # 检查有没有相邻的两个id，有的话则不是code
        for i in range(len(res) - 1):
            if res[i][0] == tokenize.NAME and res[i + 1][0] == tokenize.NAME \
                    and not iskeyword(res[i][1]) \
                    and not iskeyword(res[i + 1][1]):
                return True
        return False


class SimpleCodeTokenizer(CodeTokenizer):
    def __init__(self, vocab_file):
        self.init_vocab(vocab_file)

    def init_vocab(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)
        self.vocab = {v: k + 10 for k, v in enumerate(self.vocab)}
        self.vocab['<PAD>'] = 0  # padding position
        self.vocab['<CLS>'] = 1  # added in front of every input sample (BERT)
        self.vocab['<SEP>'] = 2  # separator token (BERT)
        self.vocab['<MSK>'] = 3  # mask token (BERT)
        self.vocab['<UID>'] = 4  # unknown_id, toknum==1
        self.vocab['<PUN>'] = 5  # punctuation, toknum==54
        self.vocab['<NUM>'] = 6  # number, toknum==2
        self.vocab['<STR>'] = 7  # string, toknum==3
        self.vocab['<SPE>'] = 8  # special character, toknum==59
        self.vocab['<COM>'] = 9 # comment, toknum==60

    def from_row_to_token_id(self, row):
        """把一行代码转成token"""
        row = row.lstrip(" #'\"").rstrip(" '\"").replace('"""', '"').replace("'''", "'")
        data_generator = create_generator([row])
        tokens_iterator = tokenize.tokenize(data_generator)
        ids = []
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                logging.debug(f"{toknum}, {tokval}")
                if toknum == tokenize.NAME:
                    ids.append(
                        self.vocab.get(tokval)) if tokval in self.vocab.keys() else ids.append(
                        self.vocab.get('<UID>'))
                elif toknum == tokenize.NUMBER:
                    ids.append(self.vocab.get('<NUM>'))
                elif toknum == tokenize.STRING:
                    ids.append(self.vocab.get('<STR>'))
                elif toknum == tokenize.NEWLINE:
                    ids.append(self.vocab.get('<SEP>'))  # \n
                elif toknum == tokenize.OP:
                    ids.append(
                        self.vocab.get(tokval)) if tokval in self.vocab.keys() else ids.append(
                        self.vocab.get('<PUN>'))
                elif toknum == tokenize.ERRORTOKEN:
                    ids.append(self.vocab.get('<SPE>'))
                elif toknum == tokenize.COMMENT:
                    ids.append(self.vocab.get('<COM>'))
        except tokenize.TokenError:
            pass  # 遍历到末尾会raise error
        return ids


class CodeSplitTokenizer(CodeTokenizer):
    def __init__(self, vocab_file):
        super().__init__(vocab_file)

    def tokenize_with_type(self, row):
        """返回切分后的token流与类型"""
        data_generator = create_generator([row])
        tokens_iterator = tokenize.tokenize(data_generator)
        tokens = []
        tokens_type = []
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                if toknum == tokenize.NAME:
                    sub_tokens = split_token(tokval)
                    for sub_token in sub_tokens:
                        tokens.append(sub_token)
                        tokens_type.append(toknum)
                else:
                    tokens.append(tokval)
                    tokens_type.append(toknum)
        except tokenize.TokenError:
            pass
        return tokens, tokens_type

    def from_row_to_token_id(self, row):
        """把一行代码转成token"""
        data_generator = create_generator([row])
        tokens_iterator = tokenize.tokenize(data_generator)
        ids = []
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                if toknum == tokenize.NAME:
                    sub_tokens = split_token(tokval)
                    for sub_token in sub_tokens:
                        ids.append(
                            self.vocab.get(sub_token)) if sub_token in self.vocab.keys() else ids.append(
                            self.vocab.get('<UID>'))
                elif toknum == tokenize.NUMBER:
                    ids.append(self.vocab.get('<NUM>'))
                elif toknum == tokenize.STRING:
                    ids.append(self.vocab.get('<STR>'))
                elif toknum == tokenize.OP:
                    ids.append(
                        self.vocab.get(tokval)) if tokval in self.vocab.keys() else ids.append(
                        self.vocab.get('<PUN>'))
                elif toknum == tokenize.ERRORTOKEN:
                    ids.append(self.vocab.get('<SPE>'))
                elif toknum == tokenize.COMMENT:
                    ids.append(self.vocab.get('<COM>'))
        except tokenize.TokenError:
            pass  # 遍历到末尾会raise error
        return ids

    @staticmethod
    def is_adjacent_id(row):
        '''屏蔽该方法，因为经过split必然会导致两个id紧靠在一起'''
        pass


class SimpleSplitCodeTokenizer(SimpleCodeTokenizer):
    def __init__(self, vocab_file):
        super().__init__(vocab_file)

    def tokenize_with_type(self, row):
        """返回切分后的token流与类型"""
        row = row.lstrip(" #'\"").rstrip(" '\"").replace('"""', '"').replace("'''", "'")
        data_generator = create_generator([row])
        tokens_iterator = tokenize.tokenize(data_generator)
        tokens = []
        tokens_type = []
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                if toknum == tokenize.NAME:
                    sub_tokens = split_token(tokval)
                    for sub_token in sub_tokens:
                        tokens.append(sub_token)
                        tokens_type.append(toknum)
                else:
                    tokens.append(tokval)
                    tokens_type.append(toknum)
        except tokenize.TokenError:
            pass
        return tokens, tokens_type

    def from_row_to_token_id(self, row):
        """把一行代码转成token"""
        row = row.lstrip(" #'\"").rstrip(" '\"").replace('"""', '"').replace("'''", "'")
        data_generator = create_generator([row])
        tokens_iterator = tokenize.tokenize(data_generator)
        ids = []
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                if toknum == tokenize.NAME:
                    sub_tokens = split_token(tokval)
                    for sub_token in sub_tokens:
                        ids.append(
                            self.vocab.get(sub_token)) if sub_token in self.vocab.keys() else ids.append(
                            self.vocab.get('<UID>'))
                elif toknum == tokenize.NUMBER:
                    ids.append(self.vocab.get('<NUM>'))
                elif toknum == tokenize.STRING:
                    ids.append(self.vocab.get('<STR>'))
                elif toknum == tokenize.NEWLINE:
                    ids.append(self.vocab.get('<SEP>'))  # \n
                elif toknum == tokenize.OP:
                    ids.append(
                        self.vocab.get(tokval)) if tokval in self.vocab.keys() else ids.append(
                        self.vocab.get('<PUN>'))
                elif toknum == tokenize.ERRORTOKEN:
                    ids.append(self.vocab.get('<SPE>'))
                elif toknum == tokenize.COMMENT:
                    ids.append(self.vocab.get('<COM>'))
        except tokenize.TokenError:
            pass  # 遍历到末尾会raise error
        return ids

    @staticmethod
    def is_adjacent_id(row):
        '''屏蔽该方法，因为经过split必然会导致两个id紧靠在一起'''
        pass


class ContextCodeTokenizer(SimpleCodeTokenizer):
    """TODO
    """
    def __init__(self, vocab_file):
        super().__init__(vocab_file)

    def from_feature_to_token_id_bta(self, before, text, after, maxlen=60):
        """
        遵循before+text+after的编码顺序
        :param feature:
        :return:
        """
        before_id = self.from_row_to_token_id(before)
        text_id = self.from_row_to_token_id(text)
        after_id = self.from_row_to_token_id(after)
        sum_id = np.concatenate([before_id, text_id, after_id])
        return pad_sequences(np.asarray([sum_id]), padding='post', value=0, maxlen=maxlen)[0]

    def from_feature_to_token_id_bat(self, before, text, after, maxlen=60):
        """
        遵循before+after+text的编码顺序
        :param feature:
        :return:
        """
        before_id = self.from_row_to_token_id(before)
        text_id = self.from_row_to_token_id(text)
        after_id = self.from_row_to_token_id(after)
        sum_id = np.concatenate([before_id, after_id, text_id])
        return pad_sequences(np.asarray([sum_id]), padding='post', value=0, maxlen=maxlen)[0]

    def from_feature_to_token_id_tba(self, before, text, after):
        """
        遵循text+before+after的编码顺序
        :param feature:
        :return:
        """
        before_id = self.from_row_to_token_id(before)
        text_id = self.from_row_to_token_id(text)
        after_id = self.from_row_to_token_id(after)
        sum_id = np.concatenate([text_id, before_id, after_id])
        return pad_sequences(np.asarray([sum_id]), padding='post', value=0, maxlen=maxlen)[0]


class ContextCodeSplitTokenizer(ContextCodeTokenizer):
    def __init__(self, vocab_file):
        super().__init__(vocab_file)

    def tokenize_with_type(self, row):
        """返回切分后的token流与类型"""
        row = row.lstrip(" #'\"").rstrip(" '\"").replace('"""', '"').replace("'''", "'")
        data_generator = create_generator([row])
        tokens_iterator = tokenize.tokenize(data_generator)
        tokens = []
        tokens_type = []
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                if toknum == tokenize.NAME:
                    sub_tokens = split_token(tokval)
                    for sub_token in sub_tokens:
                        tokens.append(sub_token)
                        tokens_type.append(toknum)
                else:
                    tokens.append(tokval)
                    tokens_type.append(toknum)
        except tokenize.TokenError:
            pass
        return tokens, tokens_type

    def from_row_to_token_id(self, row):
        """把一行代码转成token"""
        row = row.lstrip(" #'\"").rstrip(" '\"").replace('"""', '"').replace("'''", "'")
        data_generator = create_generator([row])
        tokens_iterator = tokenize.tokenize(data_generator)
        ids = []
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                if toknum == tokenize.NAME:
                    sub_tokens = split_token(tokval)
                    for sub_token in sub_tokens:
                        ids.append(
                            self.vocab.get(sub_token)) if sub_token in self.vocab.keys() else ids.append(
                            self.vocab.get('<UID>'))
                elif toknum == tokenize.NUMBER:
                    ids.append(self.vocab.get('<NUM>'))
                elif toknum == tokenize.STRING:
                    ids.append(self.vocab.get('<STR>'))
                elif toknum == tokenize.NEWLINE:
                    ids.append(self.vocab.get('<SEP>'))  # \n
                elif toknum == tokenize.OP:
                    ids.append(
                        self.vocab.get(tokval)) if tokval in self.vocab.keys() else ids.append(
                        self.vocab.get('<PUN>'))
                elif toknum == tokenize.ERRORTOKEN:
                    ids.append(self.vocab.get('<SPE>'))
                elif toknum == tokenize.COMMENT:
                    ids.append(self.vocab.get('<COM>'))
        except tokenize.TokenError:
            pass  # 遍历到末尾会raise error
        return ids

    @staticmethod
    def is_adjacent_id(row):
        '''屏蔽该方法，因为经过split必然会导致两个id紧靠在一起'''
        pass


########################################

def test_tokenizer():
    # tokenizer = CodeTokenizer("../vocabs/nosplit_keyword_vocab50000.txt")
    lines = ["for index, row in enumerate(lines):",
             "if len(int_array) >= threshold:",
             "        return sequence.pad_sequences(np.asarray(inputs), padding='post', value=0, maxlen=maxlen), indexes"]
    # print(tokenizer.from_lines_to_token_input_and_index(lines))

    tokenizer = CodeSplitTokenizer("../vocabs/split_keyword_vocab50000.txt")
    # print(tokenizer.tokenize(lines[-1]))
    g = tokenizer.from_lines_to_token_input_yield(lines)
    for line in g:
        print(line)


def test_short_line_in_dataset():
    logging.basicConfig(
        level=logging.DEBUG,
        filename='../logs/test_short_line_in_dataset_doc.log',
        filemode='w'
    )
    tokenizer = CodeTokenizer("../vocabs/nosplit_keyword_vocab50000.txt")
    df_train = pd.read_pickle('../datasets/df_train_line.tar.bz2')
    # tokenizer.from_lines_to_token_input(df_train[df_train['label'] == 0]['data'])
    tokenizer.from_lines_to_token_input(df_train[df_train['label'] == 1]['data'])

def test_eof():
    # df_train = pd.read_pickle('../datasets/df_train_line.tar.bz2', )
    tokenizer = CodeTokenizer('../vocabs/nosplit_keyword_vocab50000.txt')
    inp = ['r"""construction', 'except(']
    print(tokenizer.tokenize_with_type(inp[1]))

def test_code_split_tokenizer():
    tokenizer = SimpleSplitCodeTokenizer('../vocabs/split_keyword_vocab50000.txt')
    print(tokenizer.from_row_to_token_id("hello \nworld ' \n def # def # def"))

def test_context_code_tokenizer():
    cct = ContextCodeTokenizer('../vocabs/split_keyword_vocab50000.txt')
    before = 'tokenizer = SimpleSplitCodeTokenizer(\'../vocabs/split_keyword_vocab50000.txt\')'
    text = 'print(tokenizer.from_row_to_token_id("hello \nworld \' \n def # def # def"))'
    after = 'def test_context_code_tokenizer():'
    cct.from_feature_to_token_id_bta(before, text, after)


def main():
    logging.basicConfig(
        level=logging.DEBUG
    )
    # test_short_line_in_dataset()
    # test_eof()
    # test_code_split_tokenizer()
    test_context_code_tokenizer()
    


if __name__ == "__main__":
    main()