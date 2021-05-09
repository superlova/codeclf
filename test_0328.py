#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 16:55
# @Author  : Zyt
# @Site    : 
# @File    : test_0328.py
# @Software: PyCharm

"C:\Users\zyt\Downloads\Compressed\实验\tensorflow-master"

import os

ROOT_PATH = r'E:\code-AndroidStudy\DormitoryApp\app\src\main'
COUNT_TYPE = ['java','xml']
if __name__ == '__main__':
    lines = 0
    for filepath, dirnames, filenames in os.walk(ROOT_PATH):
        for filename in filenames:
            path = os.path.join(filepath, filename)
            type = filename.split(".")[1]
            if(type in COUNT_TYPE):
                count = len(open(path,encoding='UTF-8').readlines())
                print(path,",lines:",count)
                lines += count
    print("total count :",lines)

