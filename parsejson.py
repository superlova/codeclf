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
    file = os.path.join(result_path, 'code_warning.json')
    with open(file, 'r') as f:
        json_dict = load(f)
    csv_file = os.path.join(result_path, "code_warning.csv")
    write_csv(csv_file, json_dict)

def main():
    parse_js


if __name__ == '__main__':
    main()