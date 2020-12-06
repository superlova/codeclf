'''
Author: Zyt
Date: 2020-12-02 19:21:21
LastEditTime: 2020-12-03 09:09:43
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

    # def context_divide_naive(self, file_texts):
    #     for corpus in file_texts:
    #         text = corpus.split('\n')
    #         text = [line for line in text if len(line) > 0] # 只留下非空行
    #
    #         fsm = FSM(text)
    #         fsm.scan()
    #
    #         if len(text) == 1:
    #             for index in fsm.codes:
    #                 self.codes.append({"before": "", "text": text[0], "after": ""})
    #             for index in fsm.docs:
    #                 self.docs.append({"before": "", "text": text[0], "after": ""})
    #         elif len(text) == 2:
    #             for index in fsm.codes:
    #                 if index == 0:
    #                     self.codes.append({"before": "", "text": text[0], "after": text[1]})
    #                 elif index == len(text) - 1:
    #                     self.codes.append({"before": text[-2], "text": text[-1], "after": ""})
    #             for index in fsm.docs:
    #                 if index == 0:
    #                     self.docs.append({"before": "", "text": text[0], "after": text[1]})
    #                 elif index == len(text) - 1:
    #                     self.docs.append({"before": text[-2], "text": text[-1], "after": ""})
    #
    #         for index in fsm.codes:
    #             if index == 0:
    #                 self.codes.append({"before":"", "text":text[0], "after":text[1]})
    #             elif index == len(text) - 1:
    #                 self.codes.append({"before":text[-2], "text":text[-1], "after":""})
    #             else:
    #                 self.codes.append({"before":text[index-1], "text":text[index], "after":text[index+1]})
    #
    #         for index in fsm.docs:
    #             if index == 0:
    #                 self.docs.append({"before":"", "text":text[0], "after":text[1]})
    #             elif index == len(text) - 1:
    #                 self.docs.append({"before":text[-2], "text":text[-1], "after":""})
    #             else:
    #                 self.docs.append({"before":text[index-1], "text":text[index], "after":text[index+1]})

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
    processor = DataProcessor()
    data = processor.load_data(corpus_path)
    processor.context_divide(data, before=before, after=after)
    df = pd.DataFrame(data=processor.codes, columns=['before', 'text', 'after', 'label'])
    # df['label'] = 0
    df2 = pd.DataFrame(data=processor.docs, columns=['before', 'text', 'after', 'label'])
    # df2['label'] = 1

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


# def test_process_naive():
#     df = pd.read_pickle('../datasets/df_valid_corpus.tar.bz2')
#     file_text = df['code'][3010]
#
#     fsm = FSM(file_text.split('\n'))
#     fsm.scan()
#     fsm.pretty_print()
#     print("----------------------------")
#
#     processor = DataProcessor()
#     processor.context_divide([file_text])
#     print(processor.codes)
#     print("----------------------------")
#
#     processor = DataProcessor()
#     processor.context_divide_naive([file_text])
#     print(processor.codes)
    

def test_process():
    process('../datasets/df_train_corpus.tar.bz2', '../datasets/df_train_context.tar.bz2')
    process('../datasets/df_test_corpus.tar.bz2', '../datasets/df_test_context.tar.bz2')
    process('../datasets/df_valid_corpus.tar.bz2', '../datasets/df_valid_context.tar.bz2')

    process('../datasets/df_train_corpus.tar.bz2', '../datasets/df_train_context_2.tar.bz2', before=2, after=2)
    process('../datasets/df_test_corpus.tar.bz2', '../datasets/df_test_context_2.tar.bz2', before=2, after=2)
    process('../datasets/df_valid_corpus.tar.bz2', '../datasets/df_valid_context_2.tar.bz2', before=2, after=2)


def main():
    # test_df_head()
    logging.basicConfig(
        level=logging.INFO
    )
    test_process()
    # test_process_naive()


if __name__ == '__main__':
    main()
