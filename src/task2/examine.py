import os
import argparse
import json


def intersection(input, target):
    _input, _target = set(input), set(target) 
    intersection = _input & _target
    return len(intersection), len(_input), len(_target)


def f1_score(n_inter, n_input, n_target):
    if (n_input == 0) or (n_target == 0) or (n_inter == 0):
        return 0.0, 0.0, 0.0

    precision = n_inter / n_input
    recall = n_inter / n_target
    f1 = 2*(precision*recall)/(precision+recall)
    return precision, recall, f1


def main(params):
    answers = []
    with open(params['answer_path'], 'r', encoding='utf-8') as fin:
        for line in fin:
            answers.append(json.loads(line))

    predictions = []
    with open(params['prediction_path'], 'r', encoding='utf-8') as fin:
        for line in fin:
            predictions.append(json.loads(line))

    prediction_level = params['prediction_level']
    if (len(answers) != len(predictions)):
        status, micro_f1, macro_f1, avg_precision, avg_recall = 'Length dismatch', 0.0, 0.0, 0.0, 0.0
    else:
        precisions, recalls, f1s = [], [], []
        for x, y in zip(answers, predictions):
            if (x['input'] != y['input']):
                continue 
            
            max_precision, max_recall, max_f1 = 0.0, 0.0, 0.0
            for prediction in y['output']:
                for golden in x['output']:
                    if (golden['label'] != prediction['label']):
                        continue

                    tag_golden, tag_prediction = golden['segments'], prediction['segments']
                    if (prediction_level == 'loose'):
                        input_set, target_set = set(), set()
                        for key in tag_prediction:
                            input_set |= set(tag_prediction[key]['indices'])
                        for key in tag_golden:
                            target_set |= set(tag_golden[key]['indices'])

                        n_inter, n_input, n_target = intersection(input_set, target_set)
                        precision, recall, f1 = f1_score(n_inter, n_input, n_target)
                    
                    elif (prediction_level == 'strict'):
                        local_intersection, local_input_len, local_target_len = 0, 0, 0
                        for key in tag_prediction:
                            if (key not in tag_golden):
                                local_input_len += len(tag_prediction[key]['indices'])
                            else:
                                n_inter, n_input, n_target = intersection(tag_prediction[key]['indices'], tag_golden[key]['indices'])
                                local_intersection += n_inter
                                local_input_len += n_input
                                local_target_len += n_target

                        precision, recall, f1 = f1_score(local_intersection, local_input_len, local_target_len)

                    if (f1 > max_f1):
                        max_precision, max_recall, max_f1 = precision, recall, f1
            
            precisions.append(max_precision)
            recalls.append(max_recall)
            f1s.append(max_f1)

        status = 'Accepted'
        avg_precision = sum(precisions)/len(precisions)
        avg_recall = sum(recalls)/len(recalls)
        micro_f1 = 2*(avg_precision*avg_recall)/(avg_precision+avg_recall)
        macro_f1 = sum(f1s)/len(f1s)

    print(status)
    print('Result on %s prediction level:' %prediction_level)
    print('Micro F1 score: %f' %(micro_f1))
    print('Macro F1 score: %f' %(macro_f1))
    print('Average precision: %f' %(avg_precision))
    print('Average recall: %f' %(avg_recall))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_path', type=str, default='./data/raw/task2_new/dev.jsonlines')
    parser.add_argument('--prediction_path', type=str, default='./data/raw/task2_new/dev.jsonlines')
    parser.add_argument('--prediction_level', type=str, choices=['loose', 'medium', 'strict'], default='strict')

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    main(params)