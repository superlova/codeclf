# -*- coding:UTF-8 -*-
"""
@Description Transform json to csv.
@Author Zhang YT
@Date 2020/10/16 15:18
"""

from json import load
import os
from csv import DictWriter


def write_csv(csv_file, json_dict):
    csv_columns = ['file_path', 'lineno', 'content']
    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in json_dict['problems']:
                writer.writerow(data)
    except IOError as e:
        print("I/O error:", e)


def parse_js():
    result_path = os.path.join(os.getcwd(), 'results')
    file = os.path.join(result_path, 'commented_out_codes.json')
    with open(file, 'r') as f:
        json_dict = load(f)
    csv_file = os.path.join(result_path, "commented_out_codes.csv")
    write_csv(csv_file, json_dict)


def pretty_print(lines, lines_res):
    from colorama import init, Fore, Back, Style
    init(autoreset=True, wrap=True)
    print("py文件中的注释如下")
    print(Fore.RED + "将code打印为红色")
    print(Fore.GREEN + "docstring打印为绿色")
    for line, line_res in zip(lines, lines_res):
        if line_res:
            print(Fore.RED + line)
        else:
            print(Fore.GREEN + line)


def create_generator(data):
    def generator():
        for elem in data:
            try:
                yield str.encode(elem)
            except GeneratorExit:
                return
            except Exception as e:
                yield str.encode('')
    g = generator()  # 生成器
    def next_element():
        return next(g)
    return next_element  # 迭代器


def main():
    parse_js


if __name__ == '__main__':
    main()
