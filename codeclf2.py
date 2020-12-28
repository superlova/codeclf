'''
Date: 2020-12-28 08:54:28
LastEditors: superlova
LastEditTime: 2020-12-28 12:28:14
FilePath: /codeclf/codeclf2.py
'''

import os
from token import NAME, NUMBER, STRING, OP, ERRORTOKEN, COMMENT
from tokenize import tokenize, TokenError
from ast import parse
from json import dump
from argparse import ArgumentParser
from numpy import asarray, squeeze
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from functools import wraps

from utils.Utils import create_generator

class Classifier(object):
    def __init__(self,
                 scan_path,  # 指定扫描目录，或者文件
                 mc_path,  # （可选）训练好的character模型
                 mt_path,  # （可选）训练好的 token模型
                 vocab_path,  # （可选）token模型使用的词表文件
                 res_path,  # （可选）输出结果的目录
                 keyword="vocabs/vocab_keywords.txt"
                 ):
        self.scan_path = scan_path
        self.mc_path = mc_path
        self.mt_path = mt_path
        self.vocab_path = vocab_path
        self.res_path = res_path

        self.model_loader()
        self.init_dict(keyword)

    def model_loader(self):
        self.lstm_model_character = load_model(self.mc_path)
        self.lstm_model_token = load_model(self.mt_path)

    def init_dict(self, keyword):
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890~!@#$%^&*()` ,./<>?;':\"[]{}=-+_\t\r\n|\\"
        self.c2i = dict((c, i + 2) for i, c in enumerate(alphabet))
        self.c2i['<PAD>'] = 0
        self.c2i['<UNK>'] = 1
        self.i2c = dict((i, c) for c, i in self.c2i.items())

        vocab = []
        with open(vocab_path, 'r', encoding='utf8') as f:
            for line in f:
                vocab.append(line.rstrip('\n'))
        self.t2i = {row: index + 8 for index, row in enumerate(vocab)}
        self.t2i['<PAD>'] = 0
        self.t2i['<UNK>'] = 1
        self.t2i['<ID>'] = 2
        self.t2i['<PUN>'] = 3
        self.t2i['<NUM>'] = 4
        self.t2i['<STR>'] = 5
        self.t2i['<SPE>'] = 6
        self.t2i['<COM>'] = 7
        self.i2t = {v: k for k, v in self.t2i.items()}

        self.id_vocab = []
        with open(vocab_path, 'r', encoding='utf8') as f:
            for line in f:
                self.id_vocab.append(line.rstrip('\n'))

    @staticmethod
    def scan_subdir(scan_path):
        pyfiles = []
        scan_path = os.path.abspath(scan_path)
        for file_path, _, files in os.walk(scan_path):
            for file in files:
                if file.endswith('.py'):
                    pyfiles.append(os.path.join(file_path, file))
        return pyfiles

    @staticmethod
    def load_file(filename):
        sharps = []
        try:
            with open(filename, 'r', encoding='utf8') as f:
                for line in f:
                    sharps.append(line.strip('\n'))
        except UnicodeDecodeError:
            pass
        return sharps

    def gather_sharp_data(self, pyfiles):
        """读取指定path对应的py文件的文本，并提取其#开头的所有行"""
        sharp_data = []
        for pyfile in pyfiles:
            pycontent = self.load_file(pyfile)
            for lineno, line in enumerate(pycontent):
                if line.lstrip().startswith('#'):
                    dic = {
                        'file_path': pyfile,
                        'lineno': lineno + 1,
                        'content': line.lstrip(' #').rstrip()
                    }
                    sharp_data.append(dic)
        return sharp_data

    def from_text_to_character_input_and_index(self, text, threshold=3, maxlen=70):
        def check_dict(word):
            if word in self.c2i.keys():
                return self.c2i.get(word)
            return self.c2i.get('<unk>')

        inputs = []
        indexes = []
        for index, row in enumerate(text):
            char_array = asarray(list(row), dtype=str)
            int_array = asarray(list(map(check_dict, char_array)))
            if len(int_array) >= threshold:
                inputs.append(int_array)
                indexes.append(index)
        return sequence.pad_sequences(asarray(inputs), padding='post', value=0, maxlen=maxlen), indexes

    def from_text_to_token_id(self, row):
        """把一行代码转成token"""
        data_generator = create_generator([row])
        tokens_iterator = tokenize(data_generator)
        tokens = []
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                if toknum == NAME:
                    tokens.append(
                        self.t2i.get(tokval)) if tokval in self.t2i.keys() else tokens.append(
                        self.t2i.get('<ID>'))
                elif toknum == NUMBER:
                    tokens.append(self.t2i.get('<NUM>'))
                elif toknum == STRING:
                    tokens.append(self.t2i.get('<STR>'))
                elif toknum == OP:
                    tokens.append(
                        self.t2i.get(tokval)) if tokval in self.t2i.keys() else tokens.append(
                        self.t2i.get('<PUN>'))
                elif toknum == ERRORTOKEN:
                    tokens.append(self.t2i.get('<SPE>'))
                elif toknum == COMMENT:
                    tokens.append(self.t2i.get('<COM>'))
        except TokenError:
            pass  # 遍历到末尾会raise error
        return tokens

    def check_adjacent_id(self, row):
        data_generator = create_generator([row])
        tokens_iterator = tokenize(data_generator)
        res = []
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                res.append((toknum, tokval))
        except TokenError:
            pass
        # 检查有没有相邻的两个id，有的话则不是code
        for i in range(len(res) - 1):
            if res[i][0] == 1 \
                    and res[i + 1][0] == 1 \
                    and res[i][1] not in self.id_vocab \
                    and res[i + 1][1] not in self.id_vocab:
                return True
        return False

    def from_text_to_token_input_and_index(self, text, threshold=3, maxlen=30):
        inputs = []
        indexes = []
        for index, row in enumerate(text):
            if self.check_adjacent_id(row):
                continue
            int_array = asarray(self.from_text_to_token_id(row))
            if len(int_array) >= threshold:
                indexes.append(index)
                inputs.append(int_array)
        return sequence.pad_sequences(asarray(inputs), padding='post', value=0, maxlen=maxlen), indexes

    @staticmethod
    def reduce_sharp_by_rule(tuple_list):
        reduced_set = []  # 还需进一步判断的行
        code_set = []  # 不需进一步判断的行
        for item in tuple_list:
            try:
                text_line = item['content']
                if len(text_line.strip('=\'\"')) <= 1 \
                        or text_line == "coding=utf-8" \
                        or text_line[0].isupper() and text_line.endswith('.') \
                        or not text_line.isascii(): # TODO 在这里判断太早，应该在
                    # 出现这种特征，代表着绝不可能是代码
                    continue
                elif text_line.startswith("from ") or text_line.startswith("import ") \
                        or text_line.startswith("self.") or " = " in text_line \
                        or text_line.startswith('(') and text_line.rstrip(',').endswith(')') \
                        or text_line.startswith('[') and text_line.rstrip(',').endswith(']'):
                    compile(text_line, '<string>', 'exec')  # 尝试编译
                    code_set.append(item)
                    continue
                elif text_line.startswith("if __name__ =="):
                    # 出现这种特征，肯定是代码
                    code_set.append(item)
                    continue
                reduced_set.append(item)
            except:
                reduced_set.append(item)  # 不通过说明from语句没通过编译
        return reduced_set, code_set

    def classify(self):
        if self.scan_path.endswith('.py'):
            path = os.path.abspath(self.scan_path)
            tuple_list = self.gather_sharp_data([path])
        else:
            tuple_list = self.gather_sharp_data(self.scan_subdir(self.scan_path))
        tuple_list, code_list = self.reduce_sharp_by_rule(tuple_list)
        if len(tuple_list) <= 0:
            self.dump_res(code_list)
            return

        sharps = [x.get('content') for x in tuple_list]
        sharp_inputs, sharp_inputs_index = self.from_text_to_token_input_and_index(sharps)
        predict_label = (self.lstm_model_token.predict(sharp_inputs) > 0.5).astype("int32")
        code_item_token = []
        mask = [squeeze(predict_label) == 0]  # code
        for lineno in asarray(sharp_inputs_index)[tuple(mask)]:
            code_item_token.append(tuple_list[lineno])

        sharps = [x.get('content') for x in tuple_list]
        sharp_inputs, sharp_inputs_index = self.from_text_to_character_input_and_index(sharps)
        predict_label = (self.lstm_model_character.predict(sharp_inputs) > 0.5).astype("int32")
        code_item_char = []
        mask = [squeeze(predict_label) == 0]  # code
        for lineno in asarray(sharp_inputs_index)[tuple(mask)]:
            code_item_char.append(tuple_list[lineno])
        code_list.extend(code_item_char)
        for item in code_item_token:
            for item2 in code_item_char:
                if item.get('content') == item2.get('content') \
                        and item.get('lineno') == item2.get('lineno') \
                        and item.get('file_path') == item2.get('file_path'):
                    break
            else:
                code_list.append(item)
        print("Total number of commented code: .", len(code_list))
        self.dump_res(code_list)

    def contains_code(self, lines):
        waiting_line_index = []
        code_line_index = set()
        for index, text_line in enumerate(lines):
            try:
                if len(text_line.strip('=\'\"')) <= 1 \
                        or text_line == "coding=utf-8" \
                        or text_line[0].isupper() and text_line.endswith('.') \
                        or not text_line.isascii():  # TODO 在这里判断太早，应该在
                    # 出现这种特征，代表着绝不可能是代码
                    continue
                elif text_line.startswith("from ") or text_line.startswith("import ") \
                        or text_line.startswith("self.") or " = " in text_line \
                        or text_line.startswith('(') and text_line.rstrip(',').endswith(')') \
                        or text_line.startswith('[') and text_line.rstrip(',').endswith(']'):
                    compile(text_line, '<string>', 'exec')
                    code_line_index.add(index)
                elif text_line.startswith("if __name__ =="):
                    code_line_index.add(index)
                waiting_line_index.append(index)
            except:
                waiting_line_index.append(index)  # 不通过说明from语句没通过编译
        # 然后切分成token再输入token模型
        sharp_inputs, _ = self.from_text_to_token_input_and_index([lines[x] for x in waiting_line_index])
        predict_labels = (self.lstm_model_token.predict(sharp_inputs) > 0.5).astype("int32")
        mask = [squeeze(predict_labels) == 0][0]  # code
        for index, label in enumerate(mask):
            if label: # code
                code_line_index.add(waiting_line_index[index])
        # 最后使用character模型逐字符判断
        sharp_inputs, _ = self.from_text_to_character_input_and_index([lines[x] for x in waiting_line_index])
        predict_label = (self.lstm_model_character.predict(sharp_inputs) > 0.5).astype("int32")
        mask = [squeeze(predict_label) == 0][0]  # code
        for index, label in enumerate(mask):
            if label:  # code
                code_line_index.add(waiting_line_index[index])
        result = [False] * len(lines)
        for index in code_line_index:
            result[index] = True
        return result

    def dump_res(self, tuple_list):
        """添加一些其他信息，然后整合成code_warning.json"""
        with open(os.path.join(self.res_path, 'code_warning.json'), 'w') as f:
            dump({'problems': tuple_list}, f)


def main():
    parser = ArgumentParser(description='Check if pyfile contains commented-out code.')

    parser.add_argument(dest='scan_path', metavar='scan_path',
                        help='Check project root path')

    parser.add_argument('-mc', '--mc_path',
                        metavar='mc_path',
                        default='models/mc.hdf5',
                        dest='mc_path')

    parser.add_argument('-mt', '--mt_path',
                        metavar='mt_path',
                        default='models/mt_20000.hdf5',
                        dest='mt_path')

    parser.add_argument('-v', '--vocab',
                        metavar='vocab_path',
                        default='vocabs/vocab_20000.txt',
                        dest='vocab_path')

    parser.add_argument('-o', dest='res_path',
                        default='results',
                        help='output file path')

    args = parser.parse_args()

    args = {'scan_path': args.scan_path,
            'mc_path': args.mc_path,
            'mt_path': args.mt_path,
            'vocab_path': args.vocab_path,
            'res_path': args.res_path,
            }

    classifier = Classifier(**args)
    classifier.classify()

if __name__ == '__main__':
    main()
