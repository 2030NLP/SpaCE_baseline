import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def read_split(data_path, split, test_mode=False):
    data = []
    with open(os.path.join(data_path, '%s.jsonlines') %(split)) as fin:
        for line in fin:
            js = json.loads(line)
            data.append(js)

            if not(test_mode) and (js['output'] is None) or (js['output'] == []):
                print(line)
                input()

    return data, len(data)


# deprecated
# retain only the first segment
# def process_data(data, tokenizer, params, test_mode=False):
#     all_inputs = []
#     all_outputs = []

#     for js in data:
#         text = js['input']
#         if not(test_mode):
#             outputs = js['output']

#         tokenize_result = tokenizer(
#             text=text,
#             add_special_tokens=True,
#             padding='max_length',
#             max_length=params['seq_max_length'],
#             truncation=True,
#             return_attention_mask=True,
#             return_token_type_ids=True,
#             return_tensors='np',
#         )
#         input_ids, token_type_ids, attention_mask = tokenize_result['input_ids'][0], tokenize_result['token_type_ids'][0], tokenize_result['attention_mask'][0]

#         # single answer 
#         # TODO: multi
#         if not(test_mode):
#             # for mark in outputs:
#             if (True):
#                 mark = outputs[0]
#                 label = mark['label']
#                 segments = mark['segments']

#                 tag_nums = [2, 6, 3]

#                 if (label == 0): # text1 & text2
#                     text1_text = segments['text1']['text']
#                     text1_indices = segments['text1']['indices']
#                     text2_text = segments['text2']['text']
#                     text2_indices = segments['text2']['indices']
#                     tag_labels = np.zeros(input_ids.shape, dtype=np.int32)
#                     for i in text1_indices:
#                         tag_labels[i+1] = 1 # one pos-bias for [CLS]
#                     for i in text2_indices:
#                         tag_labels[i+1] = 2

#                 elif (label == 1): # SPE1 & SPE2
#                     tags = ['S1', 'P1', 'E1', 'S2', 'P2', 'E2']
#                     tag_labels = np.zeros(input_ids.shape, dtype=np.int32)
#                     for x, t in enumerate(tags):
#                         if t in segments:
#                             indices = segments[t]['indices']
#                             for i in indices:
#                                 tag_labels[i+1] = (x+3)

#                 elif (label == 2): # SPE
#                     tags = ['S', 'P', 'E']
#                     tag_labels = np.zeros(input_ids.shape, dtype=np.int32)
#                     for x, t in enumerate(tags):
#                         if t in segments:
#                             indices = segments[t]['indices']
#                             for i in indices:
#                                 tag_labels[i+1] = (x+9)
#                 else:
#                     print(text)
#                     continue
                
#                 all_outputs.append([tag_labels, label])

#         all_inputs.append([input_ids, token_type_ids, attention_mask])

#     input_ids = torch.tensor(np.array([x[0] for x in all_inputs]), dtype=torch.long)
#     token_type_ids = torch.tensor(np.array([x[1] for x in all_inputs]), dtype=torch.long)
#     attention_mask = torch.tensor(np.array([x[2] for x in all_inputs]), dtype=torch.long)
#     if not(test_mode):
#         tag_labels = torch.tensor(np.array([x[0] for x in all_outputs]), dtype=torch.long)
#         labels = torch.tensor(np.array([x[1] for x in all_outputs]), dtype=torch.long)

#         tensor_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, tag_labels, labels)
#         return tensor_dataset
#     else:
#         tensor_dataset = TensorDataset(input_ids, token_type_ids, attention_mask)
#         return tensor_dataset


def process_data(data, tokenizer, params, test_mode=False):
    all_inputs = []
    all_outputs = []

    for js in data:
        text = js['input']
        if not(test_mode):
            outputs = js['output']

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

        # single answer 
        # TODO: multi
        if not(test_mode):
            tag_labels = [
                np.zeros(input_ids.shape, dtype=np.int32), 
                np.zeros(input_ids.shape, dtype=np.int32), 
                np.zeros(input_ids.shape, dtype=np.int32)
            ]
            labels = [0, 0, 0]
            for mark in outputs:
                label = mark['label']
                segments = mark['segments']

                if (label == 0): # text1 & text2
                    if (labels[0] == 1):
                        continue
                    labels[0] = 1

                    text1_text = segments['text1']['text']
                    text1_indices = segments['text1']['indices']
                    text2_text = segments['text2']['text']
                    text2_indices = segments['text2']['indices']
                    # tag_labels[0] = np.zeros(input_ids.shape, dtype=np.int32)
                    for i in text1_indices:
                        tag_labels[0][i+1] = 1 # one pos-bias for [CLS]
                    for i in text2_indices:
                        tag_labels[0][i+1] = 2

                elif (label == 1): # SPE1 & SPE2
                    if (labels[1] == 1):
                        continue
                    labels[1] = 1

                    tags = ['S1', 'P1', 'E1', 'S2', 'P2', 'E2']
                    # tag_labels[1] = np.zeros(input_ids.shape, dtype=np.int32)
                    for x, t in enumerate(tags):
                        if t in segments:
                            indices = segments[t]['indices']
                            for i in indices:
                                tag_labels[1][i+1] = (x+1)

                elif (label == 2): # SPE
                    if (labels[2] == 1):
                        continue
                    labels[2] = 1

                    tags = ['S', 'P', 'E']
                    # tag_labels[2] = np.zeros(input_ids.shape, dtype=np.int32)
                    for x, t in enumerate(tags):
                        if t in segments:
                            indices = segments[t]['indices']
                            for i in indices:
                                tag_labels[2][i+1] = (x+1)
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