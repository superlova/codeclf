'''
Date: 2020-12-28 08:54:28
LastEditors: superlova
LastEditTime: 2020-12-28 16:59:32
FilePath: /codeclf/comment_checker.py
'''

import os
from token import NAME, NUMBER, STRING, OP, ERRORTOKEN, COMMENT
from tokenize import tokenize, TokenError
from json import dump
from argparse import ArgumentParser
from numpy import asarray, squeeze
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from checker_utils import parse_js, pretty_print, create_generator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class CommentChecker(object):
    def __init__(self,
                 scan_path,  # 指定扫描目录，或者文件
                 mc_path,  # （可选）训练好的character模型
                 mt_path,  # （可选）训练好的 token模型
                 vocab_path,  # （可选）token模型使用的词表文件
                 aggresive=False,
                 output=False,  # save results to csv
                 recursive=False,
                 keyword="vocabs/vocab_keywords.txt"
                 ):
        self.scan_path = scan_path
        self.mc_path = mc_path
        self.mt_path = mt_path
        self.vocab_path = vocab_path

        self.model_loader()
        self.init_dict(keyword)
        self.aggresive = aggresive
        self.output = output
        self.recursive = recursive

    def model_loader(self):
        self.mc = load_model(self.mc_path)
        self.mt = load_model(self.mt_path)

    def init_dict(self, keyword):
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890~!@#$%^&*()` ,./<>?;':\"[]{}=-+_\t\r\n|\\"
        self.c2i = dict((c, i + 2) for i, c in enumerate(alphabet))
        self.c2i['<PAD>'] = 0
        self.c2i['<UNK>'] = 1
        self.i2c = dict((i, c) for c, i in self.c2i.items())

        vocab = []
        with open(self.vocab_path, 'r', encoding='utf8') as f:
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
        with open(keyword, 'r', encoding='utf8') as f:
            for line in f:
                self.id_vocab.append(line.rstrip('\n'))

    @staticmethod
    def scan_subdir(scan_path):
        paths = []
        scan_path = os.path.abspath(scan_path)
        for file_path, _, files in os.walk(scan_path):
            for file in files:
                if file.endswith('.py'):
                    paths.append(os.path.join(file_path, file))
        return paths

    @staticmethod
    def load_file(filename):
        comments = []
        try:
            with open(filename, 'r', encoding='utf8') as f:
                for line in f:
                    comments.append(line.strip('\n'))
        except UnicodeDecodeError:
            pass
        return comments

    def read_comments(self, paths):
        """读取指定path对应的py文件的文本，并提取其#开头的所有行"""
        sharp_data = []
        for pyfile in paths:
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

    def text_2_char(self, text, threshold=3, maxlen=70):
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
            pass
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
        for i in range(len(res) - 1):
            if res[i][0] == 1 \
                    and res[i + 1][0] == 1 \
                    and res[i][1] not in self.id_vocab \
                    and res[i + 1][1] not in self.id_vocab:
                return True
        return False

    def text_2_token(self, text, threshold=3, maxlen=30):
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
    def reduce_sharp_by_rule(comment_info):
        reduced_set = []
        code_set = []
        for item in comment_info:
            try:
                text_line = item['content']
                if len(text_line.strip('=\'\"')) <= 1 \
                        or text_line == "coding=utf-8" \
                        or text_line[0].isupper() and text_line.endswith('.') \
                        or not text_line.isascii():
                    continue
                elif text_line.startswith("from ") or text_line.startswith("import ") \
                        or text_line.startswith("self.") or " = " in text_line \
                        or text_line.startswith('(') and text_line.rstrip(',').endswith(')') \
                        or text_line.startswith('[') and text_line.rstrip(',').endswith(']'):
                    compile(text_line, '<string>', 'exec')
                    code_set.append(item)
                    continue
                elif text_line.startswith("if __name__ =="):
                    code_set.append(item)
                    continue
                reduced_set.append(item)
            except:
                reduced_set.append(item)
        return reduced_set, code_set

    def check(self):
        if not os.path.exists(self.scan_path):
            print("Path or file not exists! please check input and try again.")
            return
        if self.scan_path.endswith('.py'):
            path = os.path.abspath(self.scan_path)
            comment_info = self.read_comments([path])
        elif self.recursive:
            comment_info = self.read_comments(self.scan_subdir(self.scan_path))
        else:
            print("Not a valid python file name.")
            print("If you want scan a dir, try -r mode.")
            return

        if not self.output:
            lines = []
            for dic in comment_info:
                lines.append(dic['content'])
            comment_res = self.contains_code(lines)
            pretty_print(lines, comment_res)
            return

        comment_info, cos = self.reduce_sharp_by_rule(comment_info)
        if len(comment_info) <= 0:
            self.output_res(cos)
            return

        comments = [x.get('content') for x in comment_info]
        comment_inps, comment_idx = self.text_2_token(comments)
        predict_label = (self.mt.predict(comment_inps) > 0.5).astype("int32")
        co_from_mt = []
        mask = [squeeze(predict_label) == 0]
        for lineno in asarray(comment_idx)[tuple(mask)]:
            co_from_mt.append(comment_info[lineno])

        comments = [x.get('content') for x in comment_info]
        comment_inps, comment_idx = self.text_2_char(comments)
        predict_label = (self.mc.predict(comment_inps) > 0.5).astype("int32")
        co_from_mc = []
        mask = [squeeze(predict_label) == 0]
        for lineno in asarray(comment_idx)[tuple(mask)]:
            co_from_mc.append(comment_info[lineno])
        cos.extend(co_from_mc)

        if self.aggresive == True:
            for i in co_from_mt:
                for j in co_from_mc:
                    if i.get('content') == j.get('content') \
                            and i.get('lineno') == j.get('lineno') \
                            and i.get('file_path') == j.get('file_path'):
                        break
                else:
                    cos.append(i)
        print("In total, {} Commented-out codes were checked.".format(len(cos)))
        self.output_res(cos)

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
        comment_inps, _ = self.text_2_token([lines[x] for x in waiting_line_index])
        if len(comment_inps) != 0:
            predict_labels = (self.mt.predict(comment_inps) > 0.5).astype("int32")
            mask = [squeeze(predict_labels) == 0][0]  # code
            for index, label in enumerate(mask):
                if label:  # code
                    code_line_index.add(waiting_line_index[index])
        # 最后使用character模型逐字符判断
        comment_inps, _ = self.text_2_char([lines[x] for x in waiting_line_index])
        if len(comment_inps) != 0:
            predict_label = (self.mc.predict(comment_inps) > 0.5).astype("int32")
            mask = [squeeze(predict_label) == 0][0]  # code
            for index, label in enumerate(mask):
                if label:  # code
                    code_line_index.add(waiting_line_index[index])
        result = [False] * len(lines)
        for index in code_line_index:
            result[index] = True
        return result

    def output_res(self, comment_info):
        if not os.path.exists(os.path.abspath('./results')):
            os.makedirs('./results')
        with open(os.path.join('results', 'commented_out_codes.json'), 'w') as f:
            dump({'problems': comment_info}, f)
        parse_js()


def main():
    parser = ArgumentParser(description='Check if pyfile contains commented-out code.')

    parser.add_argument(dest='scan_path', metavar='scan_path',
                        help='Python file or path contents python files.')

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

    parser.add_argument('-a', '--aggresive', dest='aggresive',
                        action='store_true',
                        help='Use aggresive mode to find more codes.')

    parser.add_argument('-o', '--output', dest='output',
                        action='store_true',
                        help='Save results to csv file.')

    parser.add_argument('-r', '--recursive', dest='recursive',
                        action='store_true',
                        help='Search path recursively.')

    args = parser.parse_args()

    args = {'scan_path': args.scan_path,
            'mc_path': args.mc_path,
            'mt_path': args.mt_path,
            'vocab_path': args.vocab_path,
            'aggresive': args.aggresive,
            'output': args.output,
            'recursive': args.recursive
            }

    comment_checker = CommentChecker(**args)
    comment_checker.check()


if __name__ == '__main__':
    main()
