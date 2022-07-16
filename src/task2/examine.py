import os
import argparse
import json


def main(params):
    answers = []
    with open(params['answer_path'], 'r', encoding='utf-8') as fin:
        for line in fin:
            answers.append(json.loads(line))

    predictions = []
    with open(params['prediction_path'], 'r', encoding='utf-8') as fin:
        for line in fin:
            predictions.append(json.loads(line))

    if (len(answers) != len(predictions)):
        type_correct, tag_correct, type_total, tag_total = 0, 0, 0, 0
        status, type_score, tag_score = 'Length dismatch', 0.0, 0.0
    else:
        type_correct, tag_correct, type_total, tag_total = 0, 0, 0, 0
        for x, y in zip(answers, predictions):
            type_total += 1
            if (x['input'] != y['input']):
                continue 
            
            golden, prediction = x['output'][0], y['output'][0]
            if (golden['label'] == prediction['label']):
                type_correct += 1

            tag_golden, tag_prediction = golden['segments'], prediction['segments']
            tag_total += len(tag_golden)
            if (golden['label'] != prediction['label']): 
                continue

            for key in tag_prediction:
                if (key not in tag_golden):
                    if (tag_prediction[key]['indices'] == []):
                        tag_correct += 1
                else:
                    if (tag_golden[key]['indices'] == tag_prediction[key]['indices']):
                        tag_correct += 1

        status = 'Accepted'
        type_score = type_correct/type_total
        tag_score = tag_correct/tag_total

    print(status)
    print('Type accuracy: %d/%d = %f' %(type_correct, type_total, type_score))
    print('Tag accuracy: %d/%d = %f' %(tag_correct, tag_total, tag_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--answer_path', type=str, default='./data/raw/task2/test.jsonlines')
    parser.add_argument('--prediction_path', type=str, default='./data/raw/task2/test.jsonlines')

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    main(params)