'''
Date: 2020-12-28 17:19:11
LastEditors: superlova
LastEditTime: 2020-12-28 17:50:22
FilePath: /codeclf/run.py
'''

import subprocess

while True:
    print("请输入需要执行的功能的编号运行本程序：")
    print("1.程序说明")
    print("2.扫描单个Python文件，将结果按照不同颜色打印至控制台")
    print("3.扫描单个Python文件，将结果保存至results目录下")
    print("4.扫描保存Python文件的目录，将结果按照不同颜色打印至控制台")
    print("5.扫描保存Python文件的目录，将结果保存至results目录下")
    print("6.退出")
    print("-----------------------")
    command = input()
    if command == "1":
        with open("README.md", "r", encoding='utf8') as f:
            c = f.read()
        print(c)
    elif command == "2":
        path = input("请输入Python文件所在的位置\n")
        test = subprocess.run(['comment_checker.exe', path, '-a'])
    elif command == "3":
        path = input("请输入Python文件所在的位置\n")
        test = subprocess.run(['comment_checker.exe', path, '-ao'])
    elif command == "4":
        path = input("请输入包含Python文件的目录\n")
        test = subprocess.run(['comment_checker.exe', path, '-ra'])
    elif command == "5":
        path = input("请输入包含Python文件的目录\n")
        test = subprocess.run(['comment_checker.exe', path, '-rao'])
    elif command == "6":
        break
