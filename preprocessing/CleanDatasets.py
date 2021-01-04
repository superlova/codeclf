# -*- coding: utf-8 -*-
# @Author  : superlova
# @Time    : 2020/11/9 11:50
# @Function: 将df_corpus转化为df_line


import re
import os
import errno
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# import sys
# project_dir = 'C:/Users/zyt/Documents/GitHub Repositories/codeclf_gui/codeclf'
# sys.path.append(project_dir)

from utils.Utils import timeout, timethis
from utils.Utils import progress

import logging


class DataProcessor(object):
    def __init__(self):
        pattern_sharp = r'(^ *#+(.*)(?:\n|$))'  # 匹配一行纯注释，不是代码后面的那种
        pattern_3doublequote = r'("""((?:.|\r?\n)*?)""")'
        pattern_3singlequote = r"('''((?:.|\r?\n)*?)''')"

        self.patterns = re.compile(r"|".join([pattern_sharp, pattern_3doublequote, pattern_3singlequote]))
        self.pattern_empty_line = re.compile(r"\n[\s| ]*(\n|$)")
        self.pattern_sharp = re.compile(pattern_sharp)

    def extract_sharp_docstring(self, code):
        """
        extract sharp docstring from code (corpus)
        :return: list of docstring
        """
        sharp_docstring = []
        docstring_iter = re.finditer(self.pattern_sharp, code)
        for match in docstring_iter:
            temp_str = match.group(1)
            temp_str = temp_str.lstrip(" #").rstrip(" \n")
            sharp_docstring.append(temp_str)
        sharp_docstring = list(filter(lambda x: len(x) > 1, sharp_docstring))
        return sharp_docstring

    @timeout(10, os.strerror(errno.ETIMEDOUT))
    def parse_code(self, string):
        """
        delete docstring and white line from code corpus
        :return: a code corpus
        """
        string = re.sub(self.patterns, "", string)
        string = string.replace("    ", "\n")
        string = string.replace("\t", "\n")
        string = re.sub(self.pattern_empty_line, "\n", string)
        return string

    @staticmethod
    def splitter_code(code):
        code_lines = code.split('\n')
        code_lines = list(map(lambda line: line.lstrip(" #").rstrip(" "), code_lines))
        code_lines = list(filter(lambda x: len(x) > 1, code_lines))
        return code_lines

    # separation and purification code and docstring from whole dataset
    def helper(self, df, saved_path, part):
        parsed_codes = []
        docstrings = []
        size = len(df)
        for i in range(len(df)):
            try:
                sharp_docstring = self.extract_sharp_docstring(df['code'][i])
                other_docstring = self.splitter_code(df['docstring'][i])

                parsed_code = self.parse_code(df['code'][i])
                splitted_code = self.splitter_code(parsed_code)

                parsed_codes.extend(splitted_code)
                docstrings.extend(sharp_docstring)
                docstrings.extend(other_docstring)

                progress(100*i/size)
            except Exception as e:
                print("code line " + str(i) + " has error " + str(e))

        res = []
        for data, label in zip(parsed_codes, np.zeros((len(parsed_codes)))):
            res.append((data, label))
        for data, label in zip(docstrings, np.ones((len(docstrings)))):
            res.append((data, label))

        res = shuffle(res, random_state=42)  # 随机打乱

        df = pd.DataFrame(columns=['data', 'label'], data=res)
        df.to_pickle(os.path.join(saved_path, f'df_{part}_line.tar.bz2'), protocol=4)


@timethis
def corpus_to_line_dataset():
    processor = DataProcessor()
    save_path = '../datasets/'

    # df_train = pd.read_pickle(os.path.join(save_path, 'df_train_corpus.tar.bz2'))
    df_test = pd.read_pickle(os.path.join(save_path, 'df_test_corpus.tar.bz2'))
    df_valid = pd.read_pickle(os.path.join(save_path, 'df_valid_corpus.tar.bz2'))

    # processor.helper(df_train, save_path, 'train')
    processor.helper(df_test, save_path, 'test')
    processor.helper(df_valid, save_path, 'valid')


def test_splitted_func():
    df = pd.read_pickle('../datasets/df_test_corpus.tar.bz2')
    processor = DataProcessor()
    # print(df['code'][0])
    # print(processor.splitter_code(processor.parse_code(df['code'][0])))
    print(processor.splitter_code(processor.parse_code(
        "			pipeline = conn			executeAfter = False")))


def test_dataset():
    df = pd.read_pickle('../datasets/df_test_line.tar.bz2')
    df = df[:10000]
    df_doc = df[df['label']==1]
    df_code = df[df['label']==0]

    df_doc.to_csv('df_test_line_doc.csv')
    df_code.to_csv('df_test_line_code.csv')


def main():
    logging.basicConfig(
        # filename='../logs/adjacent_id_code.log',
        level=logging.INFO,
        # filemode='w'
    )
    # corpus_to_line_dataset()
    # test_splitted_func()
    test_dataset()


if __name__ == '__main__':
    main()
