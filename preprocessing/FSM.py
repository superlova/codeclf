#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/2 14:45
# @Author  : Zyt
# @Site    : 
# @File    : test_1202_2.py
# @Software: PyCharm

import re
import logging


class FSM(object):
    """
    通过一遍遍历，对py文件内代码进行标注，获取该文件内代码行codes和docstring行docs的行号。
    """
    def __init__(self, text):
        self.text = text
        self.codes = []
        self.docs = []

        # empty line --> skip
        self.pattern_empty_line = re.compile(r"^(\s*)$")

        # """This is a note.""" --> docstring
        self.pattern_quotes = re.compile("|".join([r"(^\s*'''.*'''\s*$)", r'(^\s*""".*"""\s*$)']))

        # """ This is a note --> docstring
        self.pattern_double_quotes_begin = re.compile(r'^\s*"""')
        self.pattern_single_quotes_begin = re.compile(r"^\s*'''")

        # and this is also a note""" --> docstring
        self.pattern_double_quotes_end = re.compile(r'"""')
        self.pattern_single_quotes_end = re.compile(r"'''")

        # test = """hello""" --> code
        self.pattern_quotes_inner_code = re.compile("|".join([r'^\s*\S+.*""".*""".*', r"^\s*\S+.*'''.*'''.*"]))

        # test = """ hello --> code
        # but other lines are docs""" -->docstring
        self.pattern_double_quotes_outer_code = re.compile(r'^\s*\S+.*""".*')
        self.pattern_single_quotes_outer_code = re.compile(r"^\s*\S+.*'''.*")

    def START(self):
        return self.CODE

    def CODE(self, next_line_index):
        """
        event=# || event'''contains''' >>> DOCSTRING_ONE
        event=''' >>> DOCSTRING_SINGLE
        event=\"\"\" >>> DOCSTRING_DOUBLE
        还有一些corner cases
        与其说CODE状态，不如说中心状态更恰当，因为该状态下的代码可能被判断为代码，也可能被识别成docstring
        :param next_line_index:
        :return:
        """
        next_line = self.text[next_line_index]

        if next_line.lstrip(" ").startswith("#") or self.pattern_quotes.match(next_line):
            self.docs.append(next_line_index)
            return self.DOCSTRING_ONE

        elif self.pattern_single_quotes_begin.match(next_line):
            self.docs.append(next_line_index)
            return self.DOCSTRING_SINGLE

        elif self.pattern_double_quotes_begin.match(next_line):
            self.docs.append(next_line_index)
            return self.DOCSTRING_DOUBLE
        # 只有先排除inner code的情况
        elif self.pattern_quotes_inner_code.match(next_line):
            self.codes.append(next_line_index)
            return self.CODE
        # 才能正确判断outer code
        elif self.pattern_single_quotes_outer_code.match(next_line):
            self.codes.append(next_line_index)
            return self.DOCSTRING_SINGLE

        elif self.pattern_double_quotes_outer_code.match(next_line):
            self.codes.append(next_line_index)
            return self.DOCSTRING_DOUBLE

        else:
            self.codes.append(next_line_index)
            return self.CODE

    def DOCSTRING_ONE(self, next_line_index):
        return self.CODE(next_line_index)

    def DOCSTRING_DOUBLE(self, next_line_index):
        next_line = self.text[next_line_index]
        if self.pattern_double_quotes_end.search(next_line):
            self.docs.append(next_line_index)
            return self.CODE
        else:
            self.docs.append(next_line_index)
            return self.DOCSTRING_DOUBLE

    def DOCSTRING_SINGLE(self, next_line_index):
        next_line = self.text[next_line_index]
        if self.pattern_single_quotes_end.search(next_line):
            self.docs.append(next_line_index)
            return self.CODE
        else:
            self.docs.append(next_line_index)
            return self.DOCSTRING_SINGLE

    def scan(self):
        """
        逐行遍历self.text并分类
        :return:
        """
        state = self.START()
        logging.debug(f"----{state.__name__}----")
        for line_index in range(len(self.text)):
            # 空行应该全都跳过去
            if self.pattern_empty_line.match(self.text[line_index]):
                continue
            logging.debug(f"line {line_index}: {self.text[line_index]}")
            state = state(line_index)
            logging.debug(f"----{state.__name__}----")

    def pretty_print(self):
        """
        很方便地将code打印为红色，docstring打印为绿色
        :return:
        """
        mask = [0] * len(self.text)
        for id in self.docs:
            mask[id] = 1
        for index, bit in enumerate(mask):
            if bit == 0:
                print(f"\033[31m{self.text[index]}\033[0m")  # code
            else:
                print(f"\033[32m{self.text[index]}\033[0m")  # docstring


def test_DFA():
    corpus = """    def CODE(self, next_line_index):
        next_line = self.text[next_line_index]
        if next_line.lstrip(" ").startswith("#"):
            return DOCSTRING_ONE(next_line)
        test = '''else:
            self.codes.append(next_line_index)
            return CODE
        '''
    #def DOCSTRING_ONE(self, next_line_index):
    #    next_line = self.text[next_line_index]
    #    if next_line.lstrip(" ").startswith("#"):
    #        self.docs.append(next_line_index)
    #        return DOCSTRING_ONE
    #    else:
    #        return CODE(next_line_index)"""
    print(corpus)
    fsm = FSM(corpus.split('\n'))
    fsm.scan()
    print(fsm.codes)
    print(fsm.docs)


def main():
    logging.basicConfig(
        level=logging.INFO
    )
    test_DFA()


if __name__ == '__main__':
    main()
