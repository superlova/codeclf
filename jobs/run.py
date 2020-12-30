import os
import sys
import subprocess


def main():
    python_path = os.path.join(os.path.dirname(sys.executable), 'python.exe')
    print(python_path)
    tuple_list = [('bilstm_1', 'bta', '3', '3', '0.2', '', '13'),
                  ('bilstm_1', 'tba', '3', '3', '0.2', '', '14'),
                  ('bilstm_1', 'bat', '3', '3', '0.2', '', '15'),
                  ('bilstm_1', 'bta', '3', '3', '0.2', 'split', '16'),
                  ('bilstm_1', 'tba', '3', '3', '0.2', 'split', '17'),
                  ('bilstm_1', 'bat', '3', '3', '0.2', 'split', '18')
                  ]
    for row in tuple_list:
        print("\n\n=============model[{}]=============\n\n".format(row[6]))
        out_bytes = subprocess.check_output(
            [python_path, 'train1.py', '-mt', row[0], '-m', row[1], '-cb',
             row[2], '-ca', row[3], '-d', row[4], '-v', '50000', '-s', row[5],
             '-t', row[6]])
        out_text = out_bytes.decode('utf-8')
        with open('result_{}.txt'.format(row[6]), 'w') as f:
            f.write(out_text)
        print("\n\n=============end[{}]=============\n\n".format(row[6]))


if __name__ == '__main__':
    main()