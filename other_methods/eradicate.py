'''
Date: 2020-12-18 00:25:29
LastEditors: superlova
LastEditTime: 2020-12-18 00:27:32
FilePath: \codeclf\other_methods\eradicate.py
'''
"""Removes commented-out Python code."""

# from __future__ import print_function
# from __future__ import unicode_literals

import difflib
import io
import os
import re
import tokenize

__version__ = '2.0.0'


class Eradicator(object):
    """Eradicate comments."""
    BRACKET_REGEX = re.compile(r'^[()\[\]{}\s]+$')
    CODING_COMMENT_REGEX = re.compile(r'.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)')
    DEF_STATEMENT_REGEX = re.compile(r"def .+\)[\s]+->[\s]+[a-zA-Z_][a-zA-Z0-9_]*:$")
    FOR_STATEMENT_REGEX = re.compile(r"for [a-zA-Z_][a-zA-Z0-9_]* in .+:$")
    HASH_NUMBER = re.compile(r'#[0-9]')
    MULTILINE_ASSIGNMENT_REGEX = re.compile(r'^\s*\w+\s*=.*[(\[{]$')
    PARTIAL_DICTIONARY_REGEX = re.compile(r'^\s*[\'"]\w+[\'"]\s*:.+[,{]\s*$')
    PRINT_RETURN_REGEX = re.compile(r'^(print|return)\b\s*')
    WITH_STATEMENT_REGEX = re.compile(r"with .+ as [a-zA-Z_][a-zA-Z0-9_]*:$")

    CODE_INDICATORS = ['(', ')', '[', ']', '{', '}', ':', '=', '%',
                       'print', 'return', 'break', 'continue', 'import']
    CODE_KEYWORDS = [r'elif\s+.*', 'else', 'try', 'finally', r'except\s+.*']
    CODE_KEYWORDS_AGGR = CODE_KEYWORDS + [r'if\s+.*']
    WHITESPACE_HASH = ' \t\v\n#'

    DEFAULT_WHITELIST = (
        r'pylint',
        r'pyright',
        r'noqa',
        r'type:\s*ignore',
        r'fmt:\s*(on|off)',
        r'TODO',
        r'FIXME',
        r'XXX'
    )
    WHITELIST_REGEX = re.compile(r'|'.join(DEFAULT_WHITELIST), flags=re.IGNORECASE)

    def comment_contains_code(self, line, aggressive=True):
        """Return True comment contains code."""
        line = line.lstrip()
        if not line.startswith('#'):
            return False

        line = line.lstrip(self.WHITESPACE_HASH).strip()

        # Ignore non-comment related hashes. For example, "# Issue #999".
        if self.HASH_NUMBER.search(line):
            return False

        # Ignore whitelisted comments
        if self.WHITELIST_REGEX.search(line):
            return False

        if self.CODING_COMMENT_REGEX.match(line):
            return False

        # Check that this is possibly code.
        for symbol in self.CODE_INDICATORS:
            if symbol in line:
                break
        else:
            return False

        if self.multiline_case(line, aggressive=aggressive):
            return True

        for symbol in self.CODE_KEYWORDS_AGGR if aggressive else self.CODE_KEYWORDS:
            if re.match(r'^\s*' + symbol + r'\s*:\s*$', line):
                return True

        line = self.PRINT_RETURN_REGEX.sub('', line)

        if self.PARTIAL_DICTIONARY_REGEX.match(line):
            return True

        try:
            compile(line, '<string>', 'exec')
        except (SyntaxError, TypeError, UnicodeDecodeError):
            return False
        else:
            return True

    def comment_contains_code_no_sharp(self, line, aggressive=True):
        """Return True comment contains code."""
        line = line.lstrip(' #')
        # if not line.startswith('#'):
        #     return False

        line = line.lstrip(self.WHITESPACE_HASH).strip()

        # Ignore non-comment related hashes. For example, "# Issue #999".
        if self.HASH_NUMBER.search(line):
            return False

        # Ignore whitelisted comments
        if self.WHITELIST_REGEX.search(line):
            return False

        if self.CODING_COMMENT_REGEX.match(line):
            return False

        # Check that this is possibly code.
        for symbol in self.CODE_INDICATORS:
            if symbol in line:
                break
        else:
            return False

        if self.multiline_case(line, aggressive=aggressive):
            return True

        for symbol in self.CODE_KEYWORDS_AGGR if aggressive else self.CODE_KEYWORDS:
            if re.match(r'^\s*' + symbol + r'\s*:\s*$', line):
                return True

        line = self.PRINT_RETURN_REGEX.sub('', line)

        if self.PARTIAL_DICTIONARY_REGEX.match(line):
            return True

        try:
            compile(line, '<string>', 'exec')
        except (SyntaxError, TypeError, UnicodeDecodeError):
            return False
        else:
            return True


    def multiline_case(self, line, aggressive=True):
        """Return True if line is probably part of some multiline code."""
        if aggressive:
            for ending in ')]}':
                if line.endswith(ending + ':'):
                    return True

                if line.strip() == ending + ',':
                    return True

            # Check whether a function/method definition with return value
            # annotation
            if self.DEF_STATEMENT_REGEX.search(line):
                return True

            # Check weather a with statement
            if self.WITH_STATEMENT_REGEX.search(line):
                return True

            # Check weather a for statement
            if self.FOR_STATEMENT_REGEX.search(line):
                return True

        if line.endswith('\\'):
            return True

        if self.MULTILINE_ASSIGNMENT_REGEX.match(line):
            return True

        if self.BRACKET_REGEX.match(line):
            return True

        return False


    def commented_out_code_line_numbers(self, source, aggressive=True):
        """Yield line numbers of commented-out code."""
        sio = io.StringIO(source)
        try:
            for token in tokenize.generate_tokens(sio.readline):
                token_type = token[0]
                start_row = token[2][0]
                line = token[4]

                if (token_type == tokenize.COMMENT and
                        line.lstrip().startswith('#') and
                        self.comment_contains_code(line, aggressive)):
                    yield start_row
        except (tokenize.TokenError, IndentationError):
            pass


    def filter_commented_out_code(self, source, aggressive=True):
        """Yield code with commented out code removed."""
        marked_lines = list(self.commented_out_code_line_numbers(source,
                                                            aggressive))
        sio = io.StringIO(source)
        previous_line = ''
        for line_number, line in enumerate(sio.readlines(), start=1):
            if (line_number not in marked_lines or
                    previous_line.rstrip().endswith('\\')):
                yield line
            previous_line = line

    def identify_commented_out_code(self, source, aggressive=True):
        """Yield commented out code."""
        marked_lines = list(self.commented_out_code_line_numbers(source,
                                                                 aggressive))
        sio = io.StringIO(source)
        previous_line = ''
        for line_number, line in enumerate(sio.readlines(), start=1):
            if (line_number in marked_lines and not
                    previous_line.rstrip().endswith('\\')):
                yield line
            previous_line = line


    def fix_file(self, filename, args, standard_out):
        """Run filter_commented_out_code() on file."""
        encoding = self.detect_encoding(filename)
        with self.open_with_encoding(filename, encoding=encoding) as input_file:
            source = input_file.read()

        filtered_source = ''.join(self.filter_commented_out_code(source,
                                                            args.aggressive))

        if source != filtered_source:
            if args.in_place:
                with self.open_with_encoding(filename, mode='w',
                                        encoding=encoding) as output_file:
                    output_file.write(filtered_source)
            else:
                diff = difflib.unified_diff(
                    source.splitlines(),
                    filtered_source.splitlines(),
                    'before/' + filename,
                    'after/' + filename,
                    lineterm='')
                standard_out.write('\n'.join(list(diff) + ['']))
            return True


    def open_with_encoding(self, filename, encoding, mode='r'):
        """Return opened file with a specific encoding."""
        return io.open(filename, mode=mode, encoding=encoding,
                       newline='')  # Preserve line endings


    def detect_encoding(self, filename):
        """Return file encoding."""
        try:
            with open(filename, 'rb') as input_file:
                from lib2to3.pgen2 import tokenize as lib2to3_tokenize
                encoding = lib2to3_tokenize.detect_encoding(input_file.readline)[0]

                # Check for correctness of encoding.
                with self.open_with_encoding(filename, encoding) as input_file:
                    input_file.read()

            return encoding
        except (SyntaxError, LookupError, UnicodeDecodeError):
            return 'latin-1'

    def update_whitelist(self, new_whitelist, extend_default=True):
        """Updates the whitelist."""
        if extend_default:
            self.WHITELIST_REGEX = re.compile(
                r'|'.join(list(self.DEFAULT_WHITELIST) + new_whitelist),
                flags=re.IGNORECASE)
        else:
            self.WHITELIST_REGEX = re.compile(
                r'|'.join(new_whitelist),
                flags=re.IGNORECASE)

from utils.Utils import timethis
import numpy as np
import pandas as pd
from codeclf import Classifier

@timethis
def erad(source):
    eradicator = Eradicator()
    eradicator_result = []
    for line in source:
        eradicator_result.append(eradicator.comment_contains_code_no_sharp(line))
    eradicator_result = np.asarray(eradicator_result)
    return eradicator_result

@timethis
def codeclf(source):
    classifier = Classifier(
        root_path='.',  # 指定扫描目录，或者文件
        model_character_path=os.path.abspath('../models/mc.hdf5'),  # （可选）训练好的character模型
        model_token_path=os.path.abspath('../models/mt_20000.hdf5'),  # （可选）训练好的 token模型
        vocab_path=os.path.abspath('../vocabs/vocab_20000.txt'),  # （可选）token模型使用的词表文件
        outfile=os.path.abspath('../results'),  # （可选）输出结果的目录
        keyword=os.path.abspath('../vocabs/vocab_keywords.txt')
    )
    codeclf_results = np.asarray(classifier.contains_code(source.tolist()))
    return codeclf_results

def compare_result(predict_res, real_label):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for e_label, r_label in zip(predict_res, real_label):
        if e_label and r_label:
            tp += 1
        elif e_label and not r_label:
            fp += 1
        elif not e_label and r_label:
            fn += 1
        else:
            tn += 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    return acc, precision, recall, f1

def test_erdicate(path):
    df = pd.read_pickle(path)
    source = df['data']
    real_label = np.asarray(df['label'] == 0)
    print("item num:", len(source))

    eradicator_result = erad(source)
    codeclf_results = codeclf(source)

    acc_e, precision_e, recall_e, f1_e = compare_result(eradicator_result, real_label)
    print("eradicate acc: ", acc_e)
    print("eradicate precision: ", precision_e)
    print("eradicate recall: ", recall_e)
    print("eradicate f1: ", f1_e)

    acc_c, precision_c, recall_c, f1_c = compare_result(codeclf_results, real_label)
    print("codeclf acc: ", acc_c)
    print("codeclf precision: ", precision_c)
    print("codeclf recall: ", recall_c)
    print("codeclf f1: ", f1_c)


def test_classifier():
    import numpy as np
    import pandas as pd
    df_test = pd.read_pickle('../datasets/df_test_line.tar.bz2')
    source = df_test['data'][:100]

    from codeclf import  Classifier
    classifier = Classifier(
        root_path='.',  # 指定扫描目录，或者文件
        model_character_path=os.path.abspath('../models/mc.hdf5'),  # （可选）训练好的character模型
        model_token_path=os.path.abspath('../models/mt_20000.hdf5'),  # （可选）训练好的 token模型
        vocab_path=os.path.abspath('../vocabs/vocab_20000.txt'),  # （可选）token模型使用的词表文件
        outfile=os.path.abspath('../results'),  # （可选）输出结果的目录
        keyword=os.path.abspath('../vocabs/vocab_keywords.txt')
    )
    codeclf_results = classifier.contains_code(source.tolist())
    print(codeclf_results)


def main(argv, standard_out, standard_error):
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, prog='eradicate')
    parser.add_argument('-i', '--in-place', action='store_true',
                        help='make changes to files instead of printing diffs')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='drill down directories recursively')
    parser.add_argument('-a', '--aggressive', action='store_true',
                        help='make more aggressive changes; '
                             'this may result in false positives')
    parser.add_argument('-e', '--error', action="store_true",
                        help="Exit code based on result of check")
    parser.add_argument('--version', action='version',
                        version='%(prog)s ' + __version__)
    parser.add_argument('--whitelist', action="store",
                        help=(
                            'String of "#" separated comment beginnings to whitelist. '
                            'Single parts are interpreted as regex. '
                            'OVERWRITING the default whitelist: {}'
                        ).format(Eradicator.DEFAULT_WHITELIST))
    parser.add_argument('--whitelist-extend', action="store",
                        help=(
                            'String of "#" separated comment beginnings to whitelist '
                            'Single parts are interpreted as regex. '
                            'Overwrites --whitelist. '
                            'EXTENDING the default whitelist: {} '
                        ).format(Eradicator.DEFAULT_WHITELIST))
    parser.add_argument('files', nargs='+', help='files to format')

    args = parser.parse_args(argv[1:])

    eradicator = Eradicator()

    if args.whitelist_extend:
        eradicator.update_whitelist(args.whitelist_extend.split('#'), True)
    elif args.whitelist:
        eradicator.update_whitelist(args.whitelist.split('#'), False)

    filenames = list(set(args.files))
    change_or_error = False
    while filenames:
        name = filenames.pop(0)
        if args.recursive and os.path.isdir(name):
            for root, directories, children in os.walk('{}'.format(name)):
                filenames += [os.path.join(root, f) for f in children
                              if f.endswith('.py') and
                              not f.startswith('.')]
                directories[:] = [d for d in directories
                                  if not d.startswith('.')]
        else:
            try:
                change_or_error = eradicator.fix_file(name, args=args, standard_out=standard_out) or change_or_error
            except IOError as exception:
                print('{}'.format(exception), file=standard_error)
                change_or_error = True
    if change_or_error and args.error:
        return 1


if __name__ == '__main__':
    test_erdicate('../datasets/df_train_line.tar.bz2')
    test_erdicate('../datasets/df_test_line.tar.bz2')
    test_erdicate('../datasets/df_valid_line.tar.bz2')
    # test_classifier()