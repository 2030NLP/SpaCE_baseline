import json
import os

def evaluate_split(data_path, split):
    type_map = {
        'A': 0,
        'B': 1,
        'C': 2,
    }
    label_nums = [0, 0, 0]
    max_len = 0
    with open(os.path.join(data_path, 'task2_%s.jsonl') %(split)) as fin:
        for line in fin:
            js = json.loads(line)
            if (js['reasons'] is None):
                continue
            
            for reason in js['reasons']:
                type_id = type_map[reason['type']]
                label_nums[type_id] += 1
            max_len = max(max_len, len(js['context']))

    total_num = sum(label_nums)
    print('Max length on %s split: %d' %(split, max_len))
    for i in range(3):
        print('Type %d ratio on %s split: %d/%d = %f' %(i, split, label_nums[i], total_num, label_nums[i]/total_num))

if __name__ == '__main__':
    evaluate_split('../../data/input/task2', 'train')
    evaluate_split('../../data/input/task2', 'dev')
    evaluate_split('../../data/input/task2', 'test')