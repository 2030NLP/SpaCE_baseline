import json
import os

def evaluate_split(data_path, split):
    data = []
    pos_num, neg_num = 0, 0
    max_len = 0
    with open(os.path.join(data_path, '%s.jsonlines') %(split)) as fin:
        for line in fin:
            js = json.loads(line)
            if (js['output'] is None):
                continue

            if (js['output'] == 1):
                pos_num += 1
            else:
                neg_num += 1

            max_len = max(max_len, len(js['input']))

    total_num = pos_num + neg_num
    print('Max length on %s split: %d' %(split, max_len))
    print('Pos ratio on %s split: %d/%d = %f' %(split, pos_num, total_num, pos_num/total_num))
    print('Neg ratio on %s split: %d/%d = %f' %(split, neg_num, total_num, neg_num/total_num))

if __name__ == '__main__':
    evaluate_split('../../data/raw/task1', 'train')
    evaluate_split('../../data/raw/task1', 'dev')
    evaluate_split('../../data/raw/task1', 'test')