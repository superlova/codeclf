'''
Author: your name
Date: 2020-12-02 11:10:04
LastEditTime: 2020-12-03 15:25:15
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \codeclf\preprocessing\ContextEncoder.py
'''
# -*- coding: utf-8 -*-
# @Author  : superlova
# @Time    : 2020/11/24 8:56
# @Function:




def test_extract():
    text = '''
class ContextEncoder(object):
    # def __init__(self):
    #     pass

    @staticmethod
    def splitter_code(code):
        code_lines = code.split('\n')
        code_lines = list(map(lambda line: line.lstrip(" #").rstrip(" "), code_lines))
        code_lines = list(filter(lambda x: len(x) > 1, code_lines))
        return code_lines

    @staticmethod
    def encode_helper(code_lines):
        contexts = []
        for i in range(1, len(code_lines) - 1):
            contexts.append({'context_before': code_lines[i - 1],
                             'context': code_lines[i],
                             'context_after': code_lines[i + 1]})
        return contexts
    '''


def test_other():
    code = """
    hello world \ 
    end
    """
    print(code)

def main():
    # test_extract()
    test_other()


if __name__ == '__main__':
    main()
