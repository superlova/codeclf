'''
Author: your name
Date: 2020-12-06 21:32:15
LastEditTime: 2020-12-06 21:36:34
LastEditors: superlova
Description: In User Settings Edit
FilePath: \codeclf\preprocessing\CorpusEncoder.py
'''

# -*- coding: utf-8 -*-
# @Author  : superlova
# @Time    : 2020/11/24 8:56
# @Function:

import sys, os
import tokenize
import keyword
import pandas as pd

project_dir = 'C:/Users/zyt/Documents/GitHub Repositories/codeclf_gui/codeclf'
sys.path.append(os.path.join(project_dir, 'utils'))
from Utils import create_generator


class CorpusEncoder(object):
    def __init__(self):
        self.tokens = {}

    def get_context_only_id(self, corpus):
        data_generator = create_generator([corpus])
        tokens_iterator = tokenize.tokenize(data_generator)
        try:
            for toknum, tokval, _, _, _ in tokens_iterator:
                if toknum == 1 and not keyword.iskeyword(tokval):
                    if tokval in self.tokens.keys():
                        self.tokens[tokval] += 1
                    else:
                        self.tokens[tokval] = 1
        except tokenize.TokenError:
            print("EOF")
            # pass  # 遍历到末尾会raise error
        return self.tokens


def test_get_context_only_id():

    df = pd.read_pickle('../datasets/df_valid_corpus.tar.bz2')
    file_text = df['code'][3010]

    ce = CorpusEncoder()
    corpus_tokens = ce.get_context_only_id(file_text)
    print(repr(corpus_tokens))
    # sorted_tokens = sorted(corpus_tokens.items(), key=lambda x: x[1], reverse=True)


def main():
    test_get_context_only_id()


if __name__ == '__main__':
    main()
