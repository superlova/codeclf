import json
import os
import csv

def main():
    result_path = os.path.join(os.getcwd(), 'results')
    file = os.path.join(result_path, 'code_warning.json')
    with open(file, 'r') as f:
        json_dict = json.load(f)
    # print(json_dict)
    csv_file = os.path.join(result_path, "code_warning.csv")
    csv_columns = ['file', 'line', 'highlighted_element', 'offset', 'length', 'module', 'problem_class', 'entry_point', 'description']
    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in json_dict['problems']:
                writer.writerow(data)
    except IOError as e:
        print("I/O error:", e)

if __name__ == '__main__':
    main()