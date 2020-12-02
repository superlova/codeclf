# -*- coding: utf-8 -*-
# @Author  : superlova
# @Time    : 2020/11/24 8:56
# @Function:


from enum import Enum
class State(Enum):
    CODE = 0
    DOC_ONE = 1
    DOC_PREPARE_DOUBLE = 2
    DOC_PREPARE_SINGLE = 3
    SLASH = 4


def split_code(code):
    code_lines = code.split('\n')
    code_lines = list(map(lambda line: line.lstrip(" #").rstrip(" "), code_lines))
    code_lines = list(filter(lambda x: len(x) > 1, code_lines))
    return code_lines


class DFA:
    def CODE(self):
        pass

    def DOCSTRING_DOUBLE(self):
        pass

    def DOCSTRING_SINGLE(self):
        pass

    def DOCSTRING_ONE(self):
        pass

    def CODE_SLASH(self):
        pass






def extract(text, indexes):
    context = []
    if len(text) == 0:
        return ""

    for index in indexes:
        if 0 < index < len(text):
            context.append({'context_before': text[index - 1],
                            'context': text[index],
                            'context_after': text[index + 1]})
        elif index < len(text):
            context.append({'context_before': "",
                            'context': text[index],
                            'context_after': text[index + 1]})
        elif 0 < index:
            context.append({'context_before': text[index - 1],
                            'context': text[index],
                            'context_after': ""})
        else:
            context.append({'context_before': "",
                            'context': text[index],
                            'context_after': ""})

    return context


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

    text = split_code(text)
    context = extract(text, [0, 1, 2, 3])
    print(context)


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
