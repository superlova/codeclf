'''
Author: Zyt
Date: 2020-12-02 19:21:21
LastEditTime: 2020-12-06 21:59:30
LastEditors: superlova
Description: In User Settings Edit
FilePath: \codeclf\preprocessing\DataProcessor.py
'''

import pandas as pd
import logging
import sys, os
import ast

project_dir = 'C:/Users/zyt/Documents/GitHub Repositories/codeclf_gui/codeclf'
sys.path.append(os.path.join(project_dir, 'utils'))

from ContextEncoder import ContextEncoder
from CorpusEncoder import CorpusEncoder
# from ASTEncoder import ASTEncoder


class DataProcessor(object):
    def __init__(self):
        self.context_encoder = ContextEncoder()
        self.corpus_encoder = CorpusEncoder()
        # self.ast_encoder = ASTEncoder()

    def process(self, corpus):
        context_data = self.context_encoder.context_encode(corpus, before=1, after=1)
        corpus_tokens = self.corpus_encoder.get_context_only_id(corpus)
        # tree = ast.parse(corpus)
        # corpus_tree_preorder = self.ast_encoder.get_ast_preorder(tree)
        # corpus_tree_inorder = self.ast_encoder.get_ast_inorder(tree)

        print(context_data)
        print(corpus_tokens)


def test_process():
    corpus_path = '../datasets/df_test_corpus.tar.bz2'
    df_data = pd.read_pickle(corpus_path)
    data = df_data['code'][90]

    dp = DataProcessor()
    dp.process(data)


def main():
    logging.basicConfig(
        level=logging.INFO
    )
    test_process()


if __name__ == '__main__':
    main()
