import os
import argparse
import torch
import json
import random
import time
import numpy as np

from tqdm import tqdm, trange

from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer

from model import Task2TypeModel, Task2TagModel
import data
import utils


def load_model(params, model_index, base_path):
    full_path = os.path.join(base_path, 'module_%d' %model_index, 'checkpoint.bin')
    params['load_model_path'] = full_path
    if (model_index == 3): # type
        model = Task2TypeModel(params)
    else: # tag 
        tag_nums = [3, 7, 4]
        model = Task2TagModel(params, tag_nums[model_index])
    return model


def predict(
    model_set, eval_dataloader, params, device, logger,
):
    with torch.no_grad():
        for model in model_set:
            model.eval()
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

        type_results = []
        tag_results = [[], [], []]
    
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            input_ids, token_type_ids, attention_mask = batch
            
            _tag_results = []
            for i in range(3): # predict tags
                tag_prediction = model_set[i].predict(
                    input_ids, 
                    token_type_ids, 
                    attention_mask,
                )

                tag_prediction = tag_prediction.detach().cpu().numpy()
                tag_results[i].append(tag_prediction)

            # predict types
            type_prediction = model_set[3].predict(
                input_ids, 
                token_type_ids, 
                attention_mask,
            )
            type_prediction = type_prediction.detach().cpu().numpy()
            type_results.append(type_prediction)
            tag_results.append(_tag_results)

        # print(len(type_results))
        type_results = np.concatenate(type_results, axis=0)
        n = type_results.shape[0]
        for i in range(3):
            tag_results[i] = np.concatenate(tag_results[i], axis=0)

        gathered_tag_results = []
        for i in range(n):
            gathered_tag_results.append([tag_results[j][i] for j in range(3)])
        return type_results, gathered_tag_results


def gather_segments(input_text, predicted_type, predicted_tags):
    all_segments = []

    if (predicted_type[0]): # text1 & text2
        num_tags = 2
        indices = [[] for i in range(num_tags)]
        names = ['text1', 'text2']
        for i, x in enumerate(predicted_tags[0]):
            if (x >= 1) and (x <= 2):
                indices[x-1].append(i-1)
        texts = [
            ''.join([input_text[i] for i in indices[j]]) for j in range(num_tags)
        ]
        segment = []
        for i in range(num_tags):
            segment.append({
                'role': names[i],
                'text': texts[i],
                'idxes': indices[i],
            })
        all_segments.append({
            'type': 'A',
            'fragments': segment,
        })
    
    if (predicted_type[1]): # SPE1 & SPE2
        num_tags = 6
        indices = [[] for i in range(num_tags)]
        names = ['S1', 'P1', 'E1', 'S2', 'P2', 'E2']
        for i, x in enumerate(predicted_tags[1]):
            if (x >= 1) and (x <= 6):
                indices[x-1].append(i-1)
        texts = [
            ''.join([input_text[i] for i in indices[j]]) for j in range(num_tags)
        ]
        segment = []
        for i in range(num_tags):
            segment.append({
                'role': names[i],
                'text': texts[i],
                'idxes': indices[i],
            })
        all_segments.append({
            'type': 'B',
            'fragments': segment,
        })

    if (predicted_type[2]): # SPE
        num_tags = 3
        indices = [[] for i in range(num_tags)]
        names = ['S', 'P', 'E']
        for i, x in enumerate(predicted_tags[2]):
            if (x >= 1) and (x <= 3):
                indices[x-1].append(i-1)
        texts = [
            ''.join([input_text[i] for i in indices[j]]) for j in range(num_tags)
        ]
        segment = []
        for i in range(num_tags):
            segment.append({
                'role': names[i],
                'text': texts[i],
                'idxes': indices[i],
            })
        all_segments.append({
            'type': 'C',
            'fragments': segment,
        })

    return all_segments


def main(params):
    model_output_path = params['output_path']
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    logger = utils.setup_logger('SpaCET2', params['output_path'])
    logger.info("Evaluation on the %s set." %params['split'])
    tokenizer = BertTokenizer.from_pretrained(params['base_model'])

    # Init model set
    base_model_path = params['load_model_path']
    model_set = []
    for i in range(4):
        model = load_model(params, i, base_model_path)
        device = model.device
        model.to(device)
        model_set.append(model)


    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load eval data
    test_samples, test_num = data.read_split(params["data_path"], params['split'], test_mode=True)
    if (logger):
        logger.info("Read %d test samples." % test_num)

    test_tensor_data = data.process_data(
        test_samples,
        tokenizer,
        params,
        test_mode=True,
    )
    test_sampler = SequentialSampler(test_tensor_data)

    test_dataloader = DataLoader(
        test_tensor_data, sampler=test_sampler, batch_size=params['eval_batch_size']
    )

    # evaluate
    type_results, tag_results = predict(
        model_set, test_dataloader, params, device=device, logger=logger,
    )

    # print(len(test_samples))
    # print(len(tag_results))
    
    out_file_path = os.path.join(params['output_path'], '%s_prediction.jsonl' %params['split'])
    with open(out_file_path, 'w', encoding='utf-8') as fout:
        for i, js in enumerate(test_samples):
            predicted_type, predicted_tags = type_results[i], tag_results[i]
            output = gather_segments(
                js['context'], 
                predicted_type,
                # [1,1,1],
                predicted_tags,
            )
            js['reasons'] = output
            json.dump(js, fout, ensure_ascii=False)
            fout.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--data_path', type=str, default='./data/raw/task2')
    parser.add_argument('--output_path', type=str, default='./data/model/task2')
    parser.add_argument('--load_model_path', type=str, default='./data/model/task2/checkpoint.bin')
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