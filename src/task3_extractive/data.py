import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from tqdm import tqdm


element_names = {
    0: '空间实体1',
    1: '空间实体2',
    2: '事件',
    3: '事实性',
    4: '时间（文本）',
    5: '时间（标签）的参照事件',
    6: '时间（标签）',
    7: '处所',
    8: '起点',
    9: '终点',
    10: '方向',
    11: '朝向',
    12: '部件处所',
    13: '部位',
    14: '形状',
    15: '路径',
    16: '距离（文本）',
    17: '距离（标签）',
}


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


def process_trigger_data(
    data, 
    tokenizer, 
    params, 
):
    tag_map = {
        'B': 1,
        'I': 2,
        'C': 3, 
        'O': 0,
    }

    all_inputs = []
    all_outputs = []

    for js in data:
        text = js['context']
        outputs = js['outputs']

        tokenize_result = tokenizer(
            text=text,
            add_special_tokens=True,
            padding='max_length',
            max_length=params['seq_max_length'],
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='np',
            return_offsets_mapping=True,
        )
        input_ids, token_type_ids, attention_mask = tokenize_result['input_ids'][0], tokenize_result['token_type_ids'][0], tokenize_result['attention_mask'][0]
        # print(tokenizer.convert_ids_to_tokens(input_ids))
        offset_mapping = tokenize_result['offset_mapping'][0]

        all_inputs.append([input_ids, token_type_ids, attention_mask])
        tag_labels = np.zeros(input_ids.shape, dtype=np.int32)
        for tup in outputs:
            if (tup[2] is None): # 不存在事件则跳过
                continue

            idxes = tup[2]['idxes']
            start, end = idxes[0], idxes[-1] 
            for i in range(input_ids.shape[0]):
                token_start, token_end = offset_mapping[i]
                if (token_start == token_end): # special tokens
                    continue

                if (token_start == start):
                    tag_labels[i] = tag_map['B']
                elif (token_start >= start) and (token_end-1 <= end):
                    tag_labels[i] = tag_map['I']
            
        all_outputs.append(tag_labels)

        # print(text)
        # predicates = []
        # for i in range(tag_labels.shape[0]):
        #     if (tag_labels[i] != 0):
        #         predicates.append((tokenizer.convert_ids_to_tokens([input_ids[i]]), tag_labels[i]))
        # print(predicates)
        # input()


    input_ids = torch.tensor(np.array([x[0] for x in all_inputs]), dtype=torch.long)
    token_type_ids = torch.tensor(np.array([x[1] for x in all_inputs]), dtype=torch.long)
    attention_mask = torch.tensor(np.array([x[2] for x in all_inputs]), dtype=torch.long)
    tag_labels = torch.tensor(np.array(all_outputs), dtype=torch.long)
    tensor_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, tag_labels)
    return tensor_dataset
    

def to_bert_input(token_idx, null_idx):
    # segment_idx = token_idx * 0
    # mask = (token_idx != null_idx)
    segment_idx = [0 for x in token_idx]
    mask = [int(x != null_idx) for x in token_idx]
    return token_idx, segment_idx, mask


def get_pos_in_token_list(char_start_pos, char_index):
    for i, p in enumerate(char_start_pos):
        if (p == char_index) and (char_start_pos[i+1] != p): # 如果后一个token在原文中位置相同，说明本token是marker
            return i
        elif (p > char_index) or ((p == -1) and (i != 0)):
            return None

    return None


def add_markers_and_padding(
    text,
    tokenizer,
    trigger_start,
    trigger_end,
    start_token,
    end_token,
    max_seq_length,
):
    left_context, trigger, right_context = text[:trigger_start], text[trigger_start:trigger_end+1], text[trigger_end+1:]
    left_tokens = tokenizer.tokenize(left_context)
    trigger_tokens = tokenizer.tokenize(trigger)
    right_tokens = tokenizer.tokenize(right_context)

    trigger_tokens = [start_token] + trigger_tokens + [end_token]
    context_tokens = (
        left_tokens + trigger_tokens + right_tokens
    )
    context_tokens = [tokenizer.cls_token] + context_tokens + [tokenizer.sep_token]
    padding = [tokenizer.pad_token] * (max_seq_length - len(context_tokens))
    context_tokens += padding
    return context_tokens


def process_element_data(
    data, 
    tokenizer, 
    params, 
    start_token='[unused1]',
    end_token='[unused2]',
):
    max_seq_length = params['seq_max_length']
    all_outputs = []

    for js in tqdm(data, desc='Data Process'):
        text = js['context']
        outputs = js['outputs']

        for tup in outputs:
            if (tup[2] is None): # 不存在事件则跳过
                continue
            
            trigger_start, trigger_end = tup[2]['idxes'][0], tup[2]['idxes'][-1]
            context_tokens = add_markers_and_padding(
                text,
                tokenizer,
                trigger_start,
                trigger_end,
                start_token,
                end_token,
                max_seq_length,
            )

            char_start_pos = []
            for i, token in enumerate(context_tokens):
                if (token == tokenizer.cls_token) or (token == tokenizer.sep_token) or (token == tokenizer.pad_token):
                    char_start_pos.append(-1)
                else:
                    last_token, last_pos = context_tokens[i-1], char_start_pos[-1]
                    if (last_token == tokenizer.cls_token):
                        last_len = 1
                    elif (last_token == tokenizer.unk_token):
                        last_len = 1
                    elif (last_token == tokenizer.sep_token) or (last_token == tokenizer.pad_token):
                        last_len = 0
                    elif (last_token == start_token) or (last_token == end_token):
                        last_len = 0
                    else:
                        last_len = len(last_token)
                    char_start_pos.append(last_pos+last_len)


            input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
            input_ids, token_type_ids, attention_mask = to_bert_input(input_ids, tokenizer.pad_token_id)
            # print(tokenizer.convert_ids_to_tokens(input_ids))

            tag_labels = np.zeros((len(input_ids)), dtype=np.int32)
            for i in range(18):
                if (i == 2) or (i == 6) or (i == 17): # 元素为“事件”或“时间标签”或“距离标签”
                    continue
                elif (i == 3): # 事实性
                    if (tup[i] == '假'):
                        fact = 0
                    else:
                        fact = 1
                elif (tup[i] is not None):
                    try:
                        idxes = tup[i]['idxes']
                    except:
                        print(tup[i])
                        print(element_names[i])
                        quit()

                    for eid in idxes:
                        p = get_pos_in_token_list(char_start_pos, eid)
                        if (p is not None):
                            tag_labels[p] = (i+1)

                    element_text = []
                    for x in range(len(input_ids)):
                        if (tag_labels[x] == i+1):
                            element_text.append(input_ids[x])

                    # print('事件 %s 的:' %tup[2]['text'])
                    # element_text = ''.join(tokenizer.convert_ids_to_tokens(element_text))
                    # print(element_names[i], element_text)
                    # input()
            
            all_outputs.append([input_ids, token_type_ids, attention_mask, tag_labels, fact])

    input_ids = torch.tensor(np.array([x[0] for x in all_outputs]), dtype=torch.long)
    token_type_ids = torch.tensor(np.array([x[1] for x in all_outputs]), dtype=torch.long)
    attention_mask = torch.tensor(np.array([x[2] for x in all_outputs]), dtype=torch.long)
    tag_labels = torch.tensor(np.array([x[3] for x in all_outputs]), dtype=torch.long)
    fact_labels = torch.tensor(np.array([x[4] for x in all_outputs]), dtype=torch.long)
    tensor_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, tag_labels, fact_labels)
    return tensor_dataset


def process_trigger_predict_data(
    data, 
    tokenizer, 
    params, 
):
    gathered_results = {
        'input': [],
        'qid': [],
        'offset_mapping': [],
    }

    for js in data:
        text = js['context']
        qid = js['qid']

        tokenize_result = tokenizer(
            text=text,
            add_special_tokens=True,
            padding='max_length',
            max_length=params['seq_max_length'],
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='np',
            return_offsets_mapping=True,
        )
        input_ids, token_type_ids, attention_mask = tokenize_result['input_ids'][0], tokenize_result['token_type_ids'][0], tokenize_result['attention_mask'][0]
        # print(tokenizer.convert_ids_to_tokens(input_ids))
        offset_mapping = tokenize_result['offset_mapping'][0]

        gathered_results['input'].append([input_ids, token_type_ids, attention_mask])
        gathered_results['qid'].append(qid)
        gathered_results['offset_mapping'].append(offset_mapping)

    input_ids = torch.tensor(np.array([x[0] for x in gathered_results['input']]), dtype=torch.long)
    token_type_ids = torch.tensor(np.array([x[1] for x in gathered_results['input']]), dtype=torch.long)
    attention_mask = torch.tensor(np.array([x[2] for x in gathered_results['input']]), dtype=torch.long)
    tensor_dataset = TensorDataset(input_ids, token_type_ids, attention_mask)
    return_dict = {
        'dataset': tensor_dataset,
        'qid': gathered_results['qid'],
        'offset_mapping': gathered_results['offset_mapping'],
    }
    return return_dict


def process_element_predict_data(
    data,
    trigger_list,
    tokenizer, 
    params,
    start_token='[unused1]',
    end_token='[unused2]',
):
    max_seq_length = params['seq_max_length']
    special_token_id = tokenizer.convert_tokens_to_ids([start_token, end_token])
    start_token_id, end_token_id = special_token_id[0], special_token_id[1]
    
    gathered_results = {
        'input': [],
        'qid': [],
        'offset_mapping': [],
        'trigger_span': [],
    }

    for i, js in enumerate(data):
        text = js['context']
        qid = js['qid']

        tokenize_result = tokenizer(
            text=text,
            add_special_tokens=True,
            padding='max_length',
            max_length=params['seq_max_length'],
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_offsets_mapping=True,
        )
        raw_input_ids = tokenize_result['input_ids'] # 非np格式是一个一维数组
        raw_offset_mapping = tokenize_result['offset_mapping']

        triggers = trigger_list[i]
        for trigger in triggers:
            trigger_start, trigger_end = trigger # 这里拿到的位置是token编号，不是char编号
            trigger_char_span = [raw_offset_mapping[trigger_start][0], raw_offset_mapping[trigger_end][1]] # [a, b)半开区间
            
            trigger_ids = [start_token_id] + raw_input_ids[trigger_start:trigger_end+1] + [end_token_id]
            input_ids = raw_input_ids[:trigger_start] + trigger_ids + raw_input_ids[trigger_end+1:]
            input_ids = input_ids[:max_seq_length]

            # print(tokenizer.convert_ids_to_tokens(input_ids))
            # input()

            trigger_mapping = [[0, 0]] + raw_offset_mapping[trigger_start:trigger_end+1] + [[0, 0]]
            offset_mapping = raw_offset_mapping[:trigger_start] + trigger_mapping + raw_offset_mapping[trigger_end+1:]
            offset_mapping = offset_mapping[:max_seq_length]

            input_ids, token_type_ids, attention_mask = to_bert_input(input_ids, tokenizer.pad_token_id)
            
            gathered_results['input'].append([input_ids, token_type_ids, attention_mask])
            gathered_results['qid'].append(qid)
            gathered_results['trigger_span'].append(trigger_char_span)
            gathered_results['offset_mapping'].append(offset_mapping)

    input_ids = torch.tensor(np.array([x[0] for x in gathered_results['input']]), dtype=torch.long)
    token_type_ids = torch.tensor(np.array([x[1] for x in gathered_results['input']]), dtype=torch.long)
    attention_mask = torch.tensor(np.array([x[2] for x in gathered_results['input']]), dtype=torch.long)
    tensor_dataset = TensorDataset(input_ids, token_type_ids, attention_mask)

    return_dict = {
        'dataset': tensor_dataset,
        'qid': gathered_results['qid'],
        'trigger_span': gathered_results['trigger_span'],
        'offset_mapping': gathered_results['offset_mapping'],
    }
    return return_dict


if __name__ == '__main__':
    pass