import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def read_split(data_path, split, test_mode=False):
    data = []
    if not(test_mode):
        file_name = 'task2_%s.jsonl' %(split)
    else:
        file_name = 'task2_%s_input.jsonl' %(split)

    with open(os.path.join(data_path, file_name)) as fin:
        for line in fin:
            js = json.loads(line)
            data.append(js)

            if not(test_mode) and (('reasons' not in js) or (js['reasons'] == [])):
                print(line)
                input()

    return data, len(data)


def process_data(data, tokenizer, params, test_mode=False):
    all_inputs = []
    all_outputs = []

    for js in data:
        text = js['context']
        if not(test_mode):
            outputs = js['reasons']

        tokenize_result = tokenizer(
            text=text,
            add_special_tokens=True,
            padding='max_length',
            max_length=params['seq_max_length'],
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='np',
        )
        input_ids, token_type_ids, attention_mask = tokenize_result['input_ids'][0], tokenize_result['token_type_ids'][0], tokenize_result['attention_mask'][0]
        # print(tokenizer.convert_ids_to_tokens(input_ids))

        # single answer for each error type
        if not(test_mode):
            tag_labels = [
                np.zeros(input_ids.shape, dtype=np.int32), 
                np.zeros(input_ids.shape, dtype=np.int32), 
                np.zeros(input_ids.shape, dtype=np.int32),
            ]
            labels = [0, 0, 0]
            for mark in outputs:
                label = mark['type']
                segments = mark['fragments']

                if (label == 'A'): # text1 & text2
                    if (labels[0] == 1):
                        continue
                    labels[0] = 1

                    tags = {
                        'text1': 1,
                        'text2': 2,
                    }
                    for t in segments:
                        if (t['role'] in tags):
                            indices = t['idxes']
                            for i in indices:
                                tag_labels[0][i+1] = tags[t['role']] # one pos-bias for [CLS]

                elif (label == 'B'): # SPE1 & SPE2
                    if (labels[1] == 1):
                        continue
                    labels[1] = 1

                    tags = {
                        'S1': 1,
                        'P1': 2,
                        'E1': 3,
                        'S2': 4,
                        'P2': 5,
                        'E2': 6,
                    }
                    for t in segments:
                        if (t['role'] in tags):
                            indices = t['idxes']
                            for i in indices:
                                tag_labels[1][i+1] = tags[t['role']] # one pos-bias for [CLS]

                elif (label == 'C'): # SPE
                    if (labels[2] == 1):
                        continue
                    labels[2] = 1

                    tags = {
                        'S': 1,
                        'P': 2,
                        'E': 3,
                    }
                    for t in segments:
                        if (t['role'] in tags):
                            indices = t['idxes']
                            for i in indices:
                                tag_labels[2][i+1] = tags[t['role']] # one pos-bias for [CLS]
                else:
                    print(text)
                    continue
                
            all_outputs.append([tag_labels, labels])

        all_inputs.append([input_ids, token_type_ids, attention_mask])

    input_ids = torch.tensor(np.array([x[0] for x in all_inputs]), dtype=torch.long)
    token_type_ids = torch.tensor(np.array([x[1] for x in all_inputs]), dtype=torch.long)
    attention_mask = torch.tensor(np.array([x[2] for x in all_inputs]), dtype=torch.long)
    if not(test_mode):
        tag_labels = torch.tensor(np.array([x[0] for x in all_outputs]), dtype=torch.long)
        labels = torch.tensor(np.array([x[1] for x in all_outputs]), dtype=torch.long)

        tensor_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, tag_labels, labels)
        return tensor_dataset
    else:
        tensor_dataset = TensorDataset(input_ids, token_type_ids, attention_mask)
        return tensor_dataset
    

if __name__ == '__main__':
    pass