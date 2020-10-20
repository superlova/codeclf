import tokenize
import ast
import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
import time
from functools import wraps
def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.process_time()
        r = func(*args, **kwargs)
        end = time.process_time()
        print('{}.{} : {}s'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper

def createGenerator(data):
    # tokenize.tokenize需要定义一个可迭代对象来获得token

    def Generator():
        for elem in data:
            try:
                yield str.encode(elem)
            except:
                yield str.encode('')
    g = Generator()  # 生成器

    def g_fn():
        return next(g)
    return g_fn  # 迭代器


class Classifier(object):
    def __init__(self,
                 root_path,
                 model_charactor_path,
                 model_token_path,
                 vocab_path,
                 outfile,
                 use_ast
                 ):
        self.model_charactor_path = model_charactor_path
        self.model_token_path = model_token_path
        self.root_path = root_path
        self.vocab_path = vocab_path
        self.outfile = outfile
        self.use_ast = use_ast

        self.load_model()
        self.init_character_dict()
        self.init_token_dict(self.vocab_path)

    def init_character_dict(self):
        # define the raw dataset
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890~!@#$%^&*()` ,./<>?;':\"[]{}=-+_\t\r\n|\\"
        # create mapping of characters to integers (0-25) and the reverse
        self.char_to_int = dict((c, i + 2) for i, c in enumerate(alphabet))
        self.char_to_int['<pad>'] = 0
        self.char_to_int['<unk>'] = 1
        self.int_to_char = dict((i, c) for c, i in self.char_to_int.items())

    def init_token_dict(self, vocab_path):
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

        self.switch = {  # Python中没有switch case语法，用这种形式代替
            1: lambda tokens, tokval: tokens.append(
                self.token_2_id.get(tokval)) if tokval in self.token_2_id.keys() else tokens.append(
                self.token_2_id.get('<ID>')),
            2: lambda tokens, tokval: tokens.append(self.token_2_id.get('<NUM>')),
            3: lambda tokens, tokval: tokens.append(self.token_2_id.get('<STR>')),
            53: lambda tokens, tokval: tokens.append(
                self.token_2_id.get(tokval)) if tokval in self.token_2_id.keys() else tokens.append(
                self.token_2_id.get('<PUN>')),
            56: lambda tokens, tokval: tokens.append(self.token_2_id.get('<SPE>')),
            57: lambda tokens, tokval: tokens.append(self.token_2_id.get('<COM>'))
        }
        
    @timethis
    def load_model(self):
        self.lstm_model_character = tf.keras.models.load_model(self.model_charactor_path)
        self.lstm_model_token = tf.keras.models.load_model(self.model_token_path)

    def get_pyfile_path(self, root_path):
        '''获取指定目录及其子目录下所有的py文件的目录'''
        pyfiles = []
        root_path = os.path.abspath(root_path)
        for file_path, _, files in os.walk(root_path):
            for file in files:
                if file.endswith('.py'):
                    pyfiles.append(os.path.join(file_path, file))
        return pyfiles

    def read_txtfile(self, filename):
        '''读取指定path对应的py文件的文本'''
        sharps = []
        try:
            with open(filename, 'r', encoding='utf8') as f:
                for line in f:
                    sharps.append(line.strip('\n'))
        except UnicodeDecodeError as e:
            pass
        return sharps

    @timethis
    def gather_sharp_data(self, pyfiles):
        '''读取指定path对应的py文件的文本，并提取其#开头的所有行'''
        sharp_data = []
        for pyfile in pyfiles:
            pycontent = self.read_txtfile(pyfile)
            for lineno, line in enumerate(pycontent):
                if line.lstrip().startswith('#'):
                    dic = {
                        'file': pyfile, 
                        'line': lineno, 
                        'highlighted_element': line.lstrip(' #').rstrip()
                    }
                    sharp_data.append(dic)
        return sharp_data

    @timethis
    def fromTextToCharacterInputAndIndex(self, text, threshold=3, maxlen=70):
        '''输入[line]，输出适合直接学习的[input]与对应的[index]。
        目的是筛选那些长度过短的line，同时将text一次性转换的效率高一些。
        加index是为了能够溯源，防止index被打乱'''
        def check_dict(word):
            if word in self.char_to_int.keys():
                return self.char_to_int.get(word)
            return self.char_to_int.get('<unk>')
        Inputs = []
        Index = []
        for index, row in enumerate(text):
            char_array = np.asarray(list(row), dtype=str)
            int_array = np.asarray(list(map(check_dict, char_array)))
            if len(int_array) >= threshold:
                Inputs.append(int_array)
                Index.append(index)
        return sequence.pad_sequences(np.asarray(Inputs), padding='post', value=0, maxlen=maxlen), Index

    def fromTextToTokenID(self, text):
        '''把一行代码转成token。
        由于要使用python的tokenize模块，必须定义一个生成器。
        定义switch是为了避免形成连续elif，但感觉更难懂了'''
        data_generator = createGenerator(text)
        tokens_iterator = tokenize.tokenize(data_generator)
        tokens = []
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                try:
                    self.switch[toknum](tokens, tokval)
                except KeyError as e:
                    pass
        except tokenize.TokenError as e:
            pass
        return tokens

    @timethis
    def fromTextToTokenInputAndIndex(self, text, threshold=3, maxlen=30):
        '''输入[line]，输出适合直接学习的[input]与对应的[index]。
        目的是筛选那些长度过短的line，同时将text一次性转换的效率高一些。
        加index是为了能够溯源，防止index被打乱'''
        Inputs = []
        Index = []
        for index, row in enumerate(text):
            int_array = np.asarray(self.fromTextToTokenID([row]))
            if len(int_array) >= threshold:
                Index.append(index)
                Inputs.append(int_array)
        return sequence.pad_sequences(np.asarray(Inputs), padding='post', value=0, maxlen=maxlen), Index

    @timethis
    def reduce_sharpset_by_ast(self, tuple_list):
        '''输入全部[{file,line,highlighted_element}]，输出编译通过的[{file,line,highlighted_element}]'''
        reduced_set = []
        for item in tuple_list:
            try:
                # file_path, lineno, text
                text_line = item['highlighted_element']
                
                if text_line.startswith('[') and text_line.rstrip(',').endswith(']') \
                        or text_line.startswith('(') and text_line.rstrip(',').endswith(')')\
                        or len(text_line.strip('=')) < 1\
                        or text_line == "coding=utf-8"\
                        or text_line[0].isupper() and text_line.endswith('.')\
                        or not text_line.isascii():
                    continue
                #ast.parse(text_line) # 尝试编译
                reduced_set.append(item)
            except:
                pass # 不通过说明不是可执行代码
        return reduced_set

    @timethis
    def classify(self):
        '''输入全部[{file,line,highlighted_element}]，输出被怀疑为代码的[{file,line,highlighted_element}]'''
        # 获得数据
        tuple_list = self.gather_sharp_data(self.get_pyfile_path(self.root_path))
        print("all testing comment number: ", len(tuple_list))
        code_item = []
        # 先预编译一下，能通过的再进行进一步测试
        if self.use_ast == 'yes':
            tuple_list = self.reduce_sharpset_by_ast(tuple_list)
        # if len(tuple_list) <= 0:
        #     print("1:no warning code")
        #     return  # 没发现值得关注的行，提前结束
        
        # 然后切分成token再输入token模型
        ######################
        # sharps = [x.get('highlighted_element') for x in tuple_list]
        # sharp_inputs, sharp_inputs_index = self.fromTextToTokenInputAndIndex(sharps)
        # #predict_label = self.lstm_model_token.predict_classes(sharp_inputs)
        # predict_label = (self.lstm_model_token.predict(sharp_inputs) > 0.5).astype("int32")
        # code_item_token = []
        # mask = [np.squeeze(predict_label) == 0]  # code
        # for lineno in np.asarray(sharp_inputs_index)[tuple(mask)]:
        #     code_item_token.append(tuple_list[lineno])
        # print("warning code_item_token: ", len(code_item_token))
        ######################
        # if len(code_item) <= 0:
        #     print("2:no warning code")
        #     return  # 没发现值得关注的行，提前结束
        
        # 最后使用character模型逐字符判断
        # tuple_list = code_item
        ######################
        sharps = [x.get('highlighted_element') for x in tuple_list]
        sharp_inputs, sharp_inputs_index = self.fromTextToCharacterInputAndIndex(sharps)
        #predict_label = self.lstm_model_character.predict_classes(sharp_inputs)
        predict_label = (self.lstm_model_character.predict(sharp_inputs) > 0.2).astype("int32")
        code_item_char = []
        mask = [np.squeeze(predict_label) == 0]  # code
        for lineno in np.asarray(sharp_inputs_index)[tuple(mask)]:
            code_item_char.append(tuple_list[lineno])
        print("warning code_item_char: ", len(code_item_char))
        ######################
        #if len(code_item) <= 0: 
            #print("3:no text")
            #return  # 没发现值得关注的行，提前结束

        ######################
        # for item in code_item_char:
        #     for item2 in code_item_token:
        #         if item.get('highlighted_element') == item2.get('highlighted_element'):
        #             break
        #     else:
        #         # item is not in code_item_token
        #         code_item.append(item)
        ######################
        code_item.extend(code_item_char)
        # code_item = list(set(code_item_token).union(set(code_item_char)))
        print("warning comment number: ", len(code_item))
        # 保存结果
        self.dump_res(code_item)

    @timethis
    def dump_res(self, tuple_list):
        '''添加一些其他信息，然后整合成code_warning.json'''
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
        with open(os.path.join(self.outfile, 'code_warning.json'), 'w') as f:
            json.dump({'problems': tuple_list}, f)


def main():
    parser = argparse.ArgumentParser(description='Check if pyfile contains psudo-docstring.')

    parser.add_argument(dest='root_path', metavar='root_path',
                        help='Check project root path')

    parser.add_argument('-mc', '--model_charactor_path',
                        metavar='model_charactor_path',
                        default='models/mc.hdf5',
                        dest='model_charactor_path',
                        help='charactor based model path')

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

    parser.add_argument('--ast', dest='use_ast',
                        choices={'yes', 'no'}, default='yes',
                        help='whether use ast')

    args = parser.parse_args()

    args = {'root_path':args.root_path,
            'model_charactor_path':args.model_charactor_path,
            'model_token_path':args.model_token_path,
            'vocab_path':args.vocab_path,
            'outfile':args.outfile,
            'use_ast':args.use_ast}

    classifier = Classifier(**args)
    classifier.classify()


if __name__ == '__main__':
    main()