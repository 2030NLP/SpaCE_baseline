import json
import os

def evaluate_split(data_path, split):
    data = []
    label_nums = [0, 0, 0]
    max_len = 0
    with open(os.path.join(data_path, '%s.jsonlines') %(split)) as fin:
        for line in fin:
            js = json.loads(line)
            if (js['output'] is None):
                continue

            label_nums[js['output'][0]['label']] += 1
            max_len = max(max_len, len(js['input']))

    total_num = sum(label_nums)
    print('Max length on %s split: %d' %(split, max_len))
    for i in range(3):
        print('Type %d ratio on %s split: %d/%d = %f' %(i, split, label_nums[i], total_num, label_nums[i]/total_num))

if __name__ == '__main__':
    evaluate_split('../../data/raw/task2', 'train')
    evaluate_split('../../data/raw/task2', 'dev')
    evaluate_split('../../data/raw/task2', 'test')