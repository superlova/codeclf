# -*- coding:UTF-8 -*-
"""
@Description Find commented-out code in python scripts.
@Author Zhang YT
@Date 2020/10/23 14:38
"""
import os
from tokenize import tokenize, TokenError
from ast import parse
from json import dump
from argparse import ArgumentParser
from numpy import asarray, squeeze
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from time import process_time
from functools import wraps

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略警告信息，不加这一句警告贼多


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
            except:
                yield str.encode('')

    g = generator()  # 生成器

    def next_element():
        return next(g)

    return next_element  # 迭代器


class Classifier(object):
    def __init__(self,
                 root_path,  # 指定扫描目录，或者文件
                 model_character_path,  # （可选）训练好的character模型
                 model_token_path,  # （可选）训练好的 token模型
                 vocab_path,  # （可选）token模型使用的词表文件
                 outfile,  # （可选）输出结果的目录
                 keyword="vocabs/vocab_keywords.txt"
                 ):
        self.root_path = root_path
        self.model_character_path = model_character_path
        self.model_token_path = model_token_path
        self.vocab_path = vocab_path
        self.outfile = outfile
        self.load_model()  # 载入模型文件
        self.init_character_dict()  # 初始化character模型所必需的词表
        self.init_token_dict(self.vocab_path)  # 初始化token模型所必需的词表
        self.init_adjacent_dict(keyword)  # 初始化python保留字表

    def load_model(self):
        self.lstm_model_character = load_model(self.model_character_path)
        self.lstm_model_token = load_model(self.model_token_path)

    def init_character_dict(self):
        """初始化character模型所必需的词表"""
        # 所有可见字符将其映射为唯一的整数
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890~!@#$%^&*()` ,./<>?;':\"[]{}=-+_\t\r\n|\\"
        self.char_to_int = dict((c, i + 2) for i, c in enumerate(alphabet))
        self.char_to_int['<pad>'] = 0
        self.char_to_int['<unk>'] = 1
        self.int_to_char = dict((i, c) for c, i in self.char_to_int.items())

    def init_token_dict(self, vocab_path):
        """初始化token模型所必需的词表"""
        # 从词表目录读取词表文件
        vocab = []
        with open(vocab_path, 'r', encoding='utf8') as f:
            for line in f:
                vocab.append(line.rstrip('\n'))
        self.token_2_id = {row: index + 8 for index, row in enumerate(vocab)}
        self.token_2_id['<PAD>'] = 0
        self.token_2_id['<UNK>'] = 1
        self.token_2_id['<ID>'] = 2
        self.token_2_id['<PUN>'] = 3
        self.token_2_id['<NUM>'] = 4
        self.token_2_id['<STR>'] = 5
        self.token_2_id['<SPE>'] = 6
        self.token_2_id['<COM>'] = 7
        self.id_2_token = {v: k for k, v in self.token_2_id.items()}

    def init_adjacent_dict(self, vocab_path):
        """初始化python保留字表"""
        # 从词表目录读取词表文件
        self.id_vocab = []
        with open(vocab_path, 'r', encoding='utf8') as f:
            for line in f:
                self.id_vocab.append(line.rstrip('\n'))

    @staticmethod
    def get_pyfile_path(root_path):
        """获取指定目录及其子目录下所有的py文件的目录"""
        pyfiles = []
        root_path = os.path.abspath(root_path)
        for file_path, _, files in os.walk(root_path):
            for file in files:
                if file.endswith('.py'):
                    pyfiles.append(os.path.join(file_path, file))
        return pyfiles

    @staticmethod
    def read_txtfile(filename):
        """读取指定path对应的py文件的文本"""
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
            pycontent = self.read_txtfile(pyfile)
            for lineno, line in enumerate(pycontent):
                if line.lstrip().startswith('#'):
                    dic = {
                        'file': pyfile,
                        'line': lineno + 1,
                        'highlighted_element': line.lstrip(' #').rstrip()
                    }
                    sharp_data.append(dic)
        return sharp_data

    def from_text_to_character_input(self, text, threshold=3, maxlen=70):
        """输入[line]，输出适合直接学习的[input]与对应的[index]。
        threshold目的是筛选那些长度过短的line
        maxlen是对齐[input]长度，方便模型输入
        加index是为了能够溯源，防止index被打乱"""

        def check_dict(word):
            if word in self.char_to_int.keys():
                return self.char_to_int.get(word)
            return self.char_to_int.get('<unk>')

        inputs = []
        for row in text:
            char_array = asarray(list(row), dtype=str)
            int_array = asarray(list(map(check_dict, char_array)))
            if len(int_array) >= threshold:
                inputs.append(int_array)
        return sequence.pad_sequences(asarray(inputs), padding='post', value=0, maxlen=maxlen)

    def from_text_to_character_input_and_index(self, text, threshold=3, maxlen=70):
        """输入[line]，输出适合直接学习的[input]与对应的[index]。
        threshold目的是筛选那些长度过短的line
        maxlen是对齐[input]长度，方便模型输入
        加index是为了能够溯源，防止index被打乱"""

        def check_dict(word):
            if word in self.char_to_int.keys():
                return self.char_to_int.get(word)
            return self.char_to_int.get('<unk>')

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
                if toknum == 1:
                    tokens.append(
                        self.token_2_id.get(tokval)) if tokval in self.token_2_id.keys() else tokens.append(
                        self.token_2_id.get('<ID>'))
                elif toknum == 2:
                    tokens.append(self.token_2_id.get('<NUM>'))
                elif toknum == 3:
                    tokens.append(self.token_2_id.get('<STR>'))
                elif toknum == 53:
                    tokens.append(
                        self.token_2_id.get(tokval)) if tokval in self.token_2_id.keys() else tokens.append(
                        self.token_2_id.get('<PUN>'))
                elif toknum == 56:
                    tokens.append(self.token_2_id.get('<SPE>'))
                elif toknum == 57:
                    tokens.append(self.token_2_id.get('<COM>'))
        except TokenError:
            pass  # 遍历到末尾会raise error
        return tokens

    def check_adjacent_id(self, row):
        """检查有没有相邻的两个id"""
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

    def from_text_to_token_input(self, text, threshold=3, maxlen=30):
        """输入[line]，输出适合直接学习的[input]与对应的[index]。
        threshold目的是筛选那些长度过短的line
        maxlen是对齐[input]长度，方便模型输入
        加index是为了能够溯源，防止index被打乱"""
        inputs = []
        for row in text:
            # 筛选那些相邻的id，2代表单词表外的id
            if self.check_adjacent_id(row):
                continue
            int_array = asarray(self.from_text_to_token_id(row))
            if len(int_array) >= threshold:
                inputs.append(int_array)
        return sequence.pad_sequences(asarray(inputs), padding='post', value=0, maxlen=maxlen)

    def from_text_to_token_input_and_index(self, text, threshold=3, maxlen=30):
        """输入[line]，输出适合直接学习的[input]与对应的[index]。
        threshold目的是筛选那些长度过短的line
        maxlen是对齐[input]长度，方便模型输入
        加index是为了能够溯源，防止index被打乱"""
        inputs = []
        indexes = []
        for index, row in enumerate(text):
            # 筛选那些相邻的id，2代表单词表外的id
            if self.check_adjacent_id(row):
                continue
            int_array = asarray(self.from_text_to_token_id(row))
            if len(int_array) >= threshold:
                indexes.append(index)
                inputs.append(int_array)
        return sequence.pad_sequences(asarray(inputs), padding='post', value=0, maxlen=maxlen), indexes

    @staticmethod
    def reduce_sharp_by_rule(tuple_list):
        """输入全部[{file,line,highlighted_element}]，
        输出符合规则的[{file,line,highlighted_element}]"""
        reduced_set = []  # 还需进一步判断的行
        code_set = []  # 不需进一步判断的行
        for item in tuple_list:
            try:
                text_line = item['highlighted_element']
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
                    # 出现这种特征，是代码的可能性大，需要经过一遍编译
                    # 通过编译则为代码，不通过则录入reduced_set
                    parse(text_line)  # 尝试编译
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

    @timethis
    def classify(self):
        """输入全部[{file,line,highlighted_element}]，
        输出被怀疑为代码的[{file,line,highlighted_element}]"""
        # 获得数据
        if self.root_path.endswith('.py'):
            path = os.path.abspath(self.root_path)
            tuple_list = self.gather_sharp_data([path])
        else:
            tuple_list = self.gather_sharp_data(self.get_pyfile_path(self.root_path))
        print(f"All testing comment number from {self.root_path}: {len(tuple_list)}.")

        # 依照确定性算法，将注释分为需要进一步判断的tuple_list和code_list
        tuple_list, code_list = self.reduce_sharp_by_rule(tuple_list)
        # 防止模型输入为空
        if len(tuple_list) <= 0:
            print("1: No commented-out code.")
            # 保存结果
            self.dump_res(code_list)
            return  # 没发现值得进一步分析的行，提前结束
        else:
            print("Commented code number find by pure grammar checker: ", len(code_list))

        # 然后切分成token再输入token模型
        sharps = [x.get('highlighted_element') for x in tuple_list]
        sharp_inputs, sharp_inputs_index = self.from_text_to_token_input_and_index(sharps)
        predict_label = (self.lstm_model_token.predict(sharp_inputs) > 0.5).astype("int32")
        code_item_token = []
        mask = [squeeze(predict_label) == 0]  # code
        for lineno in asarray(sharp_inputs_index)[tuple(mask)]:
            code_item_token.append(tuple_list[lineno])
        print("Commented code number find by `token` model: ", len(code_item_token))

        # 最后使用character模型逐字符判断
        sharps = [x.get('highlighted_element') for x in tuple_list]
        sharp_inputs, sharp_inputs_index = self.from_text_to_character_input_and_index(sharps)
        predict_label = (self.lstm_model_character.predict(sharp_inputs) > 0.5).astype("int32")
        code_item_char = []
        mask = [squeeze(predict_label) == 0]  # code
        for lineno in asarray(sharp_inputs_index)[tuple(mask)]:
            code_item_char.append(tuple_list[lineno])
        print("Commented code number find by `character` model: ", len(code_item_char))

        code_list.extend(code_item_char)
        # 两个集合取并集
        for item in code_item_token:
            for item2 in code_item_char:
                if item.get('highlighted_element') == item2.get('highlighted_element') \
                        and item.get('line') == item2.get('line') \
                        and item.get('file') == item2.get('file'):
                    break
            else:
                code_list.append(item)
        print("Total number of commented code: .", len(code_list))
        # 保存结果
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
                    # 出现这种特征，是代码的可能性大，需要经过一遍编译
                    # 通过编译则为代码，不通过则录入reduced_set
                    parse(text_line)  # 尝试编译
                    # compile(text_line, '<string>', 'exec')
                    code_line_index.add(index)
                elif text_line.startswith("if __name__ =="):
                    # 出现这种特征，肯定是代码
                    code_line_index.add(index)
                waiting_line_index.append(index)
            except:
                waiting_line_index.append(index)  # 不通过说明from语句没通过编译
        # 然后切分成token再输入token模型
        sharp_inputs = self.from_text_to_token_input([lines[x] for x in waiting_line_index])
        predict_labels = (self.lstm_model_token.predict(sharp_inputs) > 0.5).astype("int32")
        mask = [squeeze(predict_labels) == 0][0]  # code
        for index, label in enumerate(mask):
            if label: # code
                code_line_index.add(waiting_line_index[index])
        # 最后使用character模型逐字符判断
        sharp_inputs = self.from_text_to_character_input([lines[x] for x in waiting_line_index])
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
        for dic in tuple_list:
            dic['offset'] = 0
            dic['length'] = 0
            dic['module'] = ''
            dic['problem_class'] = {
                'name': '8_2',
                'severity': '',
                'inspection_name': '8_2',
                'attribute_key': ''
            }
            dic['entry_point'] = {
                'TYPE': '',
                'FQNAME': ''
            }
            dic['description'] = 'Do not use comment lines to make the code invalid.'
        with open(os.path.join(self.outfile, 'commented_out_codes.json'), 'w') as f:
            dump({'problems': tuple_list}, f)


def main():
    parser = ArgumentParser(description='Check if pyfile contains commented-out code.')

    parser.add_argument(dest='root_path', metavar='root_path',
                        help='Check project root path')

    parser.add_argument('-mc', '--model_character_path',
                        metavar='model_character_path',
                        default='models/mc.hdf5',
                        dest='model_character_path',
                        help='character based model path')

    parser.add_argument('-mt', '--model_token_path',
                        metavar='model_token_path',
                        default='models/mt_20000.hdf5',
                        dest='model_token_path',
                        help='token based model path')

    parser.add_argument('-v', '--vocab',
                        metavar='vocab_path',
                        default='vocabs/vocab_20000.txt',
                        dest='vocab_path',
                        help='token vocabulary path')

    parser.add_argument('-o', dest='outfile',
                        default='results',
                        help='output file path')

    args = parser.parse_args()

    args = {'root_path': args.root_path,
            'model_character_path': args.model_character_path,
            'model_token_path': args.model_token_path,
            'vocab_path': args.vocab_path,
            'outfile': args.outfile,
            }

    classifier = Classifier(**args)
    classifier.classify()

if __name__ == '__main__':
    main()
