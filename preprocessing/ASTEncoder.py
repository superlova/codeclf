'''
Author: your name
Date: 2020-12-06 14:40:08
LastEditTime: 2020-12-06 20:55:53
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \codeclf\preprocessing\ASTEncoder.py
'''

import ast
import pandas as pd

from graphviz import Digraph
from astpretty import pprint


class ASTEncoder(object):
    def __init__(self):
        self.inorder = []
        self.preorder = []

    def preorder_helper(self, root):
        if root is None:
            return
        self.preorder.append(str(type(root).__name__))
        nodes = ast.iter_child_nodes(root)
        for node in nodes:
            self.preorder_helper(node)

    def get_ast_preorder(self, corpus):
        ast_tree = ast.parse(corpus)
        self.preorder_helper(ast_tree)
        return self.preorder

    def get_ast_inorder(self, corpus):
        ast_tree = ast.parse(corpus)
        for node in ast.walk(ast_tree):
            self.inorder.append(type(node).__name__)
        return self.inorder

    def visit_graphviz(self, node, nodes, pindex, graph):
        """
        将node绘制出来，然后连接node与他的父节点，并递归地对孩子们执行visit
        :param node: ast对象
        :param nodes: 储存了已经绘制出来的节点编号
        :param pindex: 父节点编号
        :param graph: graphviz对象
        :return:
        """
        name = str(type(node).__name__) # 获取节点类型
        index = len(nodes) # 获取节点编号
        nodes.append(index)
        graph.node(str(index), name) # 将node绘制出来
        if index != pindex:
            graph.edge(str(index), str(pindex)) # 然后连接node与他的父节点
        for n in ast.iter_child_nodes(node):
            self.visit_graphviz(n, nodes, index, graph) # 并递归地对孩子们执行visit

    def visualize_tree(self, corpus, output_path):
        """
        使用graphviz可视化ast图形
        :param corpus:
        :return:
        """
        graph = Digraph(format='pdf')
        tree = ast.parse(corpus)
        self.visit_graphviz(tree, [], 0, graph)
        graph.render(output_path)

    def print_tree(self, corpus):
        print(pprint(ast.parse(corpus)))

################ test ###################

def test_ASTEncoder():
    df = pd.read_pickle('../datasets/df_valid_corpus.tar.bz2')
    file_text = df['code'][19]
    ae = ASTEncoder()
    ae.get_ast_preorder(file_text)


def test_find_file():
    df = pd.read_pickle('../datasets/df_valid_corpus.tar.bz2')
    print(df['code'][19])


def test_visualize():
    df = pd.read_pickle('../datasets/df_valid_corpus.tar.bz2')
    file_text  = df['code'][19]
    print(file_text)

    ae = ASTEncoder()
    ae.visualize_tree(file_text, "../results/test.txt")
    ae.get_ast_preorder(file_text)


def test_print_tree():
    df = pd.read_pickle('../datasets/df_valid_corpus.tar.bz2')
    file_text = df['code'][19]
    print(file_text)

    ae = ASTEncoder()
    ae.print_tree(file_text)


def test_order():
    df = pd.read_pickle('../datasets/df_valid_corpus.tar.bz2')
    file_text = df['code'][19]
    print(file_text)
    print("------------")
    
    tree = ast.parse(file_text)
    ae = ASTEncoder()
    print(ae.get_ast_preorder(tree))
    print(ae.get_ast_inorder(tree))


def main():
    # test_ASTEncoder()
    # test_find_file()
    test_visualize()
    # test_print_tree()
    # test_order()


if __name__ == '__main__':
    main()