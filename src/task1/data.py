import json
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

def read_split(data_path, split, test_mode=False):
    data = []
    with open(os.path.join(data_path, '%s.jsonlines') %(split)) as fin:
        for line in fin:
            js = json.loads(line)
            if not(test_mode) and (js['output'] is None):
                continue

            data.append(js)

    return data, len(data)


def process_data(data, tokenizer, params, test_mode=False):
    all_text = []
    all_labels = []

    for js in data:
        text = js['input']
        all_text.append(text)

        if not(test_mode):
            label = js['output']
            all_labels.append(label)

    tokenize_result = tokenizer(
        text=all_text,
        add_special_tokens=True,
        padding='max_length',
        max_length=params['seq_max_length'],
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='pt',
    )
    input_ids, token_type_ids, attention_mask = tokenize_result['input_ids'], tokenize_result['token_type_ids'], tokenize_result['attention_mask']
    # print(tokenizer.convert_ids_to_tokens(input_ids[0]))
    # input()
    if (test_mode):
        tensor_dataset = TensorDataset(input_ids, token_type_ids, attention_mask)
    else:
        labels = torch.tensor(all_labels, dtype=torch.long)
        tensor_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, labels)
    return tensor_dataset


if __name__ == '__main__':
    pass