import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset

def read_split(data_path, split, test_mode=False):
    data = []
    if not(test_mode):
        file_name = 'task3_%s.jsonl' %(split)
    else:
        file_name = 'task3_%s_input.jsonl' %(split)

    with open(os.path.join(data_path, file_name)) as fin:
        for line in fin:
            js = json.loads(line)
            data.append(js)

            if not(test_mode) and (('outputs' not in js) or (js['outputs'] == [])):
                print(line)
                input()

    return data


def deserialize(
    examples,
    tokenizer,
    start_token='<extra_id_0>',
    end_token='<extra_id_1>', 
    sep_token='<extra_id_2>',
):
    print(examples[0])
    input()
    all_deserialized_examples = []
    return None


def serialize_tuple(
    tuple, 
    tokenizer,
    start_token,
    end_token, 
    sep_token,
):
    elements = []
    for i in range(18):
        element = tuple[i]
        if (element is None):
            element_tokens = ' '
        elif (i == 0) or (i == 1):
            element_tokens = ' %s ' %(' '.join([str(x) for x in element['idxes']]))
        elif (type(element) == str):
            element_tokens = ' %s ' %(element)
        else:
            element_tokens = ' %s ' %(' '.join(element['text']))

        elements.append(element_tokens)

    all_element_tokens = sep_token.join(elements)
    result_str = '%s %s %s' %(start_token, all_element_tokens, end_token)
    return result_str


def process_data(
    data, 
    tokenizer, 
    params, 
    test_mode=False,
    start_token='<extra_id_0>',
    end_token='<extra_id_1>', 
    sep_token='<extra_id_2>',
):
    all_inputs = []
    all_input_text = []
    all_qids = []
    all_outputs = []

    for js in data:
        text = js['context']
        qid = js['qid']
        if not(test_mode):
            outputs = js['outputs']

        tokenize_result = tokenizer(
            text=text,
            add_special_tokens=True,
            padding='max_length',
            max_length=params['seq_max_length'],
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors='np',
        )
        input_ids = tokenize_result['input_ids'][0]
        # print(tokenizer.convert_ids_to_tokens(input_ids))
        # input()

        if not(test_mode):
            all_serialized_tuples = []
            for tup in outputs:
                result_str = serialize_tuple(
                    tup,
                    tokenizer,
                    start_token,
                    end_token, 
                    sep_token,
                )
                all_serialized_tuples.append(result_str)
                
            output_str = ' '.join(all_serialized_tuples)
            print(output_str)
            input()
            output_tokenize_result = tokenizer(
                text=output_str,
                add_special_tokens=True,
                padding='max_length',
                max_length=params['seq_max_length'],
                truncation=True,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_tensors='np',
            )
            output_ids = output_tokenize_result['input_ids'][0]
            all_outputs.append(output_ids)

        all_inputs.append(input_ids)
        all_input_text.append(text)
        all_qids.append(qid)

    input_ids = torch.tensor(all_inputs, dtype=torch.long)
    if not(test_mode):
        labels = torch.tensor(all_outputs, dtype=torch.long)

        dataset = Dataset.from_dict({
            'qids': all_qids,
            'input_ids': input_ids,
            'labels': labels,
        })
        return dataset
    else:
        dataset = {
            'qids': all_qids,
            'text': all_input_text,
            'input_ids': input_ids,
        }
        return dataset
    

if __name__ == '__main__':
    pass