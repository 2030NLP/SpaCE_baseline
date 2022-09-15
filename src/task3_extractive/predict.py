import os
import argparse
import torch
import json
import random
import time
import numpy as np

from tqdm import tqdm, trange

from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer

from model import Task3TriggerModel, Task3ElementModel
import data
import utils


def load_model(params, base_path, trigger):
    if (trigger):
        full_path = os.path.join(base_path, 'task3_extractive_trigger', 'checkpoint.bin')
        params['load_model_path'] = full_path
        model = Task3TriggerModel(params, 4)
    else:
        full_path = os.path.join(base_path, 'task3_extractive_element', 'checkpoint.bin')
        params['load_model_path'] = full_path
        model = Task3ElementModel(params, 19)
    return model


def gather_triggers(tag_result):
    tag_map = {
        'B': 1,
        'I': 2,
        'C': 3, 
        'O': 0,
    }
    trigger_list = []
    triggered = False
    start, end = None, None
    for i, x in enumerate(tag_result):
        if (x == tag_map['O']):
            triggered = False
            if (start is not None) and (end is not None):
                trigger_list.append((start, end))
                start, end = None, None
        if (x == tag_map['B']):
            triggered = True
            if (start is not None) and (end is not None):
                trigger_list.append((start, end))
            start, end = i, i
        if (x == tag_map['I']) or (x == tag_map['C']):
            if not(triggered):
                continue
            end = i

    if (start is not None) and (end is not None):
        trigger_list.append((start, end))

    return trigger_list


def predict_triggers(
    model_set, eval_dataloader, device,
):
    with torch.no_grad():
        for model in model_set:
            model_set[model].eval()
        iter_ = tqdm(eval_dataloader, desc="Trigger Prediction")

        tag_results = []
    
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            input_ids, token_type_ids, attention_mask = batch
            
            _tag_results = []
            # predict triggers
            tag_prediction = model_set['trigger'].predict(
                input_ids, 
                token_type_ids, 
                attention_mask,
            )

            tag_prediction = tag_prediction.detach().cpu().numpy()
            tag_results.append(tag_prediction)

        tag_results = np.concatenate(tag_results, axis=0)
        n = tag_results.shape[0]
        triggers = []
        for i in range(n):
            trigger_spans = gather_triggers(tag_results[i])
            triggers.append(trigger_spans)
        return triggers


def gather_elements(tag_result, fact_result):
    predicted_tuple = [None for i in range(18)]
    for i, x in enumerate(tag_result):
        if (x != 0):
            if (predicted_tuple[x-1] is None):
                predicted_tuple[x-1] = []
            predicted_tuple[x-1].append(i)
    
    if (fact_result == 0):
        predicted_tuple[3] = '假'

    return predicted_tuple


def predict_elements(
    model_set, 
    eval_dataloader, 
    device,
):
    with torch.no_grad():
        for model in model_set:
            model_set[model].eval()
        iter_ = tqdm(eval_dataloader, desc="Element Prediction")

        tag_results = []
        fact_results = []
    
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            input_ids, token_type_ids, attention_mask = batch
            
            _tag_results = []
            # predict triggers
            tag_prediction, fact_prediction = model_set['element'].predict(
                input_ids, 
                token_type_ids, 
                attention_mask,
            )

            tag_prediction = tag_prediction.detach().cpu().numpy()
            fact_prediction = fact_prediction.detach().cpu().numpy()
            tag_results.append(tag_prediction)
            fact_results.append(fact_prediction)

        tag_results = np.concatenate(tag_results, axis=0)
        fact_results = np.concatenate(fact_results, axis=0)
        n = tag_results.shape[0]
        elements = []
        for i in range(n):
            elements.append(gather_elements(tag_results[i], fact_results[i]))
        return elements


def convert_trigger_spans(test_samples, trigger_results, offset_mappings):
    converted_trigger_spans = []
    for i, sample in enumerate(test_samples):
        text = sample['context']
        trigger_result = trigger_results[i]
        offset_mapping = offset_mappings[i]

        for trigger in trigger_result:
            trigger_start, trigger_end = trigger
            trigger_char_span = [offset_mapping[trigger_start][0], offset_mapping[trigger_end][1]] # [a, b)半开区间
            converted_trigger_spans.append((trigger_char_span[0], trigger_char_span[1]))
            
            # print(text[trigger_char_span[0]:trigger_char_span[1]])
            # input()

    return converted_trigger_spans


def convert_to_tuple(
    converted_trigger_spans,
    qids,
    element_offset_mapping,
    predicted_elements,
):
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

    converted_tuples = []

    for element_tup, trigger_span, qid, offset_mapping in zip(predicted_elements, converted_trigger_spans, qids, element_offset_mapping):
        converted_tuple = [None for i in range(18)]
        for i, x in enumerate(element_tup):
            if (i == 2):
                trigger_start, trigger_end = trigger_span
                converted_tuple[2] = {
                    'text': '',
                    'idxes': list(range(trigger_start, trigger_end)),
                }
            elif (i == 3):
                converted_tuple[i] = x
            elif (x is not None):
                converted_tuple[i] = {
                    'text': '',
                    'idxes': [],
                }
                for token_idx in x:
                    offset_range = offset_mapping[token_idx]
                    for p in range(offset_range[0], offset_range[1]):
                        converted_tuple[i]['idxes'].append(p)

        converted_tuples.append((qid, converted_tuple))

    return converted_tuples


def decorate_tuples(converted_tuples, test_samples):
    qid_map = {}
    for sample in test_samples:
        qid = sample['qid']
        context = sample['context']
        qid_map[qid] = {
            'qid': qid,
            'context': context, 
            'outputs': [],
        }

    for qid, converted_tuple in converted_tuples:
        context = qid_map[qid]['context']
        for element in converted_tuple:
            if (element is None) or (type(element) == str):
                continue

            element['text'] = ''.join([context[p] for p in element['idxes']])
        qid_map[qid]['outputs'].append(converted_tuple)

    decorated_tuples = []
    for qid in qid_map:
        decorated_tuples.append(qid_map[qid])

    return decorated_tuples


def main(params):
    model_output_path = params['output_path']
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    logger = utils.setup_logger('SpaCET3', params['output_path'])
    logger.info("Evaluation on the %s set." %params['split'])
    tokenizer = AutoTokenizer.from_pretrained(params['base_model'])

    # Init model set
    base_model_path = params['load_model_path']
    model_set = {}
    for module in ['trigger', 'element']:
        model = load_model(params, base_model_path, (module == 'trigger'))
        device = model.device
        model.to(device)
        model_set[module] = model


    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load eval data
    test_samples = data.read_split(params["data_path"], params['split'], test_mode=True)

    trigger_data_dict = data.process_trigger_predict_data(
        test_samples,
        tokenizer,
        params,
    )
    trigger_tensor_data = trigger_data_dict['dataset']
    trigger_sampler = SequentialSampler(trigger_tensor_data)

    trigger_dataloader = DataLoader(
        trigger_tensor_data, sampler=trigger_sampler, batch_size=params['eval_batch_size']
    )

    trigger_results = predict_triggers(
        model_set, 
        trigger_dataloader,
        device=device,
    )

    converted_trigger_spans = convert_trigger_spans(test_samples, trigger_results, trigger_data_dict['offset_mapping'])

    element_data_dict = data.process_element_predict_data(
        test_samples,
        trigger_results,
        tokenizer,
        params,
    )
    element_tensor_data = element_data_dict['dataset']
    element_sampler = SequentialSampler(element_tensor_data)

    element_dataloader = DataLoader(
        element_tensor_data, sampler=element_sampler, batch_size=params['eval_batch_size']
    )

    element_results = predict_elements(
        model_set, 
        element_dataloader, 
        device=device
    )
    
    gathered_tuples = convert_to_tuple(
        converted_trigger_spans=converted_trigger_spans,
        qids=element_data_dict['qid'],
        element_offset_mapping=element_data_dict['offset_mapping'],
        predicted_elements=element_results,
    )

    decorated_json = decorate_tuples(gathered_tuples, test_samples)

    out_file_path = os.path.join(params['output_path'], '%s_prediction.jsonl' %params['split'])
    with open(out_file_path, 'w', encoding='utf-8') as fout:
        for i, js in enumerate(decorated_json):
            fout.write(json.dumps(js, ensure_ascii=False))
            fout.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--data_path', type=str, default='./data/raw/task3')
    parser.add_argument('--output_path', type=str, default='./data/model/task3')
    parser.add_argument('--load_model_path', type=str, default='./data/model')
    parser.add_argument('--base_model', type=str, default='hfl/chinese-bert-wwm-ext')
    
    # model arguments
    parser.add_argument('--seq_max_length', type=int, default=256)
    parser.add_argument('--cuda', action='store_true')

    # evaluation arguments
    parser.add_argument('--split', type=str, default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    main(params)