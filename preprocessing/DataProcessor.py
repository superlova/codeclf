'''
Author: Zyt
Date: 2020-12-02 19:21:21
LastEditTime: 2020-12-02 21:57:44
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \codeclf\preprocessing\DataProcessor.py
'''

import pandas as pd
import logging
import sys, os


project_dir = 'C:/Users/zyt/Documents/GitHub Repositories/codeclf_gui/codeclf'
sys.path.append(os.path.join(project_dir, 'utils'))

from FSM import FSM
from Utils import timethis


class DataProcessor(object):
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

    def context_divide(self, file_texts):
        for corpus in file_texts:
            text = corpus.split('\n')

            fsm = FSM(text)
            fsm.scan()
            if len(text) == 0:
                continue
            elif len(text) == 1:
                for index in fsm.codes:
                    self.codes.append({"before": "", "text": text[0], "after": ""})
                for index in fsm.docs:
                    self.docs.append({"before": "", "text": text[0], "after": ""})
            elif len(text) == 2:
                for index in fsm.codes:
                    if index == 0:
                        self.codes.append({"before": "", "text": text[0], "after": text[1]})
                    elif index == len(text) - 1:
                        self.codes.append({"before": text[-2], "text": text[-1], "after": ""})
                for index in fsm.docs:
                    if index == 0:
                        self.docs.append({"before": "", "text": text[0], "after": text[1]})
                    elif index == len(text) - 1:
                        self.docs.append({"before": text[-2], "text": text[-1], "after": ""})

            for index in fsm.codes:
                if index == 0:
                    self.codes.append({"before":"", "text":text[0], "after":text[1]})
                elif index == len(text) - 1:
                    self.codes.append({"before":text[-2], "text":text[-1], "after":""})
                else:
                    self.codes.append({"before":text[index-1], "text":text[index], "after":text[index+1]})

            for index in fsm.docs:
                if index == 0:
                    self.docs.append({"before":"", "text":text[0], "after":text[1]})
                elif index == len(text) - 1:
                    self.docs.append({"before":text[-2], "text":text[-1], "after":""})
                else:
                    self.docs.append({"before":text[index-1], "text":text[index], "after":text[index+1]})


@timethis
def process(corpus_path, output_path):
    processor = DataProcessor()
    data = processor.load_data(corpus_path)
    processor.context_divide(data)
    df = pd.DataFrame(data=processor.codes, columns=['before', 'text', 'after', 'label'])
    df['label'] = 0
    df2 = pd.DataFrame(data=processor.docs, columns=['before', 'text', 'after', 'label'])
    df2['label'] = 1

    print(f"df_code:{len(df)}, df_docs:{len(df2)}")
    df = pd.concat([df, df2], ignore_index=True)
    df.to_pickle(output_path, protocol=4)


def test_df_head():
    # processor = DataProcessor()
    # processor.\
    df = pd.read_pickle('../datasets/df_valid_corpus.tar.bz2')
    file_text = df['code'][3010]
    print(file_text)

    fsm = FSM(file_text.split('\n'))
    fsm.scan()
    fsm.pretty_print()


def test_process():
    process('../datasets/df_train_corpus.tar.bz2', '../datasets/df_train_context.tar.bz2')
    # process('../datasets/df_test_corpus.tar.bz2', '../datasets/df_test_context.tar.bz2')
    # process('../datasets/df_valid_corpus.tar.bz2', '../datasets/df_valid_context.tar.bz2')


def main():
    # test_df_head()
    logging.basicConfig(
        level=logging.INFO
    )
    test_process()


if __name__ == '__main__':
    main()