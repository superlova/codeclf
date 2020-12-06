'''
Author: your name
Date: 2020-12-02 11:10:04
LastEditTime: 2020-12-06 21:51:45
LastEditors: superlova
Description: In User Settings Edit
FilePath: \codeclf\preprocessing\ContextEncoder.py
'''
# -*- coding: utf-8 -*-
# @Author  : superlova
# @Time    : 2020/11/24 8:56
# @Function:

import pandas as pd
import logging
import sys, os


project_dir = 'C:/Users/zyt/Documents/GitHub Repositories/codeclf_gui/codeclf'
sys.path.append(os.path.join(project_dir, 'utils'))

from FSM import FSM
from Utils import timethis


class ContextEncoder(object):
    def __init__(self):
        self.codes = []
        self.docs = []

    def load_data(self, path : "bz2 files"):
        df = pd.read_pickle(path)
        return df['code']

    def naive_divide(self, file_texts):
        """
        直接将每行文本分类，而不带任何上下文信息
        :param file_texts:
        :return:
        """
        for corpus in file_texts:
            text = corpus.split('\n')
            fsm = FSM(text)
            fsm.scan()
            self.codes.extend((text[index] for index in fsm.codes))
            self.docs.extend((text[index] for index in fsm.docs))

    def context_divide(self, file_texts, before=1, after=1):
        for corpus in file_texts:
            # 逐个扫描每个py文件，标记每行的类别
            text = corpus.split('\n')
            text = [line for line in text if len(line) > 0] # 只留下非空行
            length = len(text)
            fsm = FSM(text)
            fsm.scan()

            for index in fsm.codes:
                # 每一行代码都查找它的上下文：
                context_before = []
                context_after = []
                if index < before: # index不够大导致before不够
                    for i in range(index):
                        context_before.append(text[i])
                else:
                    for i in range(index - before, index):
                        context_before.append(text[i])
                if length - index - 1 < after: # index过大导致after不够
                    for i in range(index + 1, length):
                        context_after.append(text[i])
                else:
                    for i in range(index + 1, index + after + 1):
                        context_after.append(text[i])

                item = {'before': context_before, 'text': text[index], 'after': context_after, 'label': 0}

                self.codes.append(item)

            for index in fsm.docs:
                # 每一行代码都查找它的上下文：
                context_before = []
                context_after = []
                if index < before:  # index不够大导致before不够
                    for i in range(index):
                        context_before.append(text[i])
                else:
                    for i in range(index - before, index):
                        context_before.append(text[i])
                if length - index - 1 < after:  # index过大导致after不够
                    for i in range(index + 1, length):
                        context_after.append(text[i])
                else:
                    for i in range(index + 1, index + after + 1):
                        context_after.append(text[i])

                item = {'before': context_before, 'text': text[index], 'after': context_after, 'label': 1}

                self.docs.append(item)


@timethis
def process(corpus_path, output_path, before=1, after=1):
    df_data = pd.read_pickle(corpus_path)
    data = df_data['code']

    processor = ContextEncoder()
    processor.context_divide(data, before=before, after=after)
    df_codes = pd.DataFrame(data=processor.codes, columns=['before', 'text', 'after', 'label'])
    df_docs = pd.DataFrame(data=processor.docs, columns=['before', 'text', 'after', 'label'])

    print(f"df_codes:{len(df_codes)}, df_docs:{len(df_docs)}")
    df = pd.concat([df_codes, df_docs], ignore_index=True)
    df.to_pickle(output_path, protocol=4)
    

def test_process():
    process('../datasets/df_train_corpus.tar.bz2', '../datasets/df_train_context.tar.bz2')
    process('../datasets/df_test_corpus.tar.bz2', '../datasets/df_test_context.tar.bz2')
    process('../datasets/df_valid_corpus.tar.bz2', '../datasets/df_valid_context.tar.bz2')

    process('../datasets/df_train_corpus.tar.bz2', '../datasets/df_train_context_2.tar.bz2', before=2, after=2)
    process('../datasets/df_test_corpus.tar.bz2', '../datasets/df_test_context_2.tar.bz2', before=2, after=2)
    process('../datasets/df_valid_corpus.tar.bz2', '../datasets/df_valid_context_2.tar.bz2', before=2, after=2)


def main():
    
    logging.basicConfig(
        level=logging.INFO
    )
    test_process()


if __name__ == '__main__':
    main()
