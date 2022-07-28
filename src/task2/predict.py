import os
import argparse
import torch
import json
import random
import time
import numpy as np

from tqdm import tqdm, trange

from torch.utils.data import DataLoader, SequentialSampler

from model import Task2Model
import data
import utils


def load_model(params):
    return Task2Model(params)


def predict(
    model, eval_dataloader, params, device, logger,
):
    with torch.no_grad():
        model.eval()
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

        type_results = []
        tag_results = []
    
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            input_ids, token_type_ids, attention_mask = batch
                
            type_prediction, tag_prediction = model.predict(
                input_ids, 
                token_type_ids, 
                attention_mask,
            )

            type_prediction = type_prediction.detach().cpu().numpy()
            tag_prediction = tag_prediction.detach().cpu().numpy()
            type_results.append(type_prediction)
            tag_results.append(tag_prediction)

        # print(len(type_results))
        type_results = np.concatenate(type_results, axis=0)
        tag_results = np.concatenate(tag_results, axis=0)
        return type_results, tag_results


def gather_segments(input_text, predicted_type, predicted_tags):
    all_segments = []

    if (predicted_type[0]): # text1 & text2
        text1_indices, text2_indices = [], []
        for i, x in enumerate(predicted_tags[0]):
            if (x == 1):
                text1_indices.append(i-1) # one pos-bias for [CLS]
            elif (x == 2):
                text2_indices.append(i-1)
        text1_text = ''.join([input_text[i] for i in text1_indices])
        text2_text = ''.join([input_text[i] for i in text2_indices])
        segment = {
            'text1': {'text': text1_text, 'indices': text1_indices},
            'text2': {'text': text2_text, 'indices': text2_indices},
        }
        all_segments.append({
            'label': 0,
            'segments': segment,
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
        segment = {}
        for i in range(num_tags):
            segment[names[i]] = {
                'text': texts[i],
                'indices': indices[i],
            }
        all_segments.append({
            'label': 1,
            'segments': segment,
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
        segment = {}
        for i in range(num_tags):
            segment[names[i]] = {
                'text': texts[i],
                'indices': indices[i],
            }
        all_segments.append({
            'label': 2,
            'segments': segment,
        })

    return all_segments


def main(params):
    model_output_path = params['output_path']
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    logger = utils.setup_logger('SpaCET2', params['output_path'])
    logger.info("Evaluation on the %s set." %params['split'])

    # Init model
    model = Task2Model(params)
    tokenizer = model.tokenizer
    device = model.device

    model = model.to(model.device)

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
        model, test_dataloader, params, device=device, logger=logger,
    )
    
    out_file_path = os.path.join(params['output_path'], '%s_prediction.jsonlines' %params['split'])
    with open(out_file_path, 'w', encoding='utf-8') as fout:
        for i, js in enumerate(test_samples):
            predicted_type, predicted_tags = type_results[i], tag_results[i]
            output = gather_segments(
                js['input'], 
                predicted_type, 
                predicted_tags,
            )
            js['output'] = output
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