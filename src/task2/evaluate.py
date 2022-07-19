import os
import argparse
import torch
import json
import random
import time
import numpy as np

from tqdm import tqdm, trange

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from model import Task2Model
import data
import utils


def load_model(params):
    return Task2Model(params)


def evaluate(
    model, eval_dataloader, params, device, logger,
):
    with torch.no_grad():
        model.eval()
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

        results = {}

        type_accuracy = 0.0
        tag_accuracy = 0.0
        nb_eval_examples = 0
        nb_eval_steps = 0
    
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            input_ids, token_type_ids, attention_mask, tag_labels, labels = batch
                
            type_prediction, tag_prediction = model.predict(
                input_ids, 
                token_type_ids, 
                attention_mask,
            )

            type_prediction = type_prediction.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            tag_prediction = tag_prediction.detach().cpu().numpy()
            tag_labels = tag_labels.detach().cpu().numpy()
            correct_type_num, correct_tag_num = utils.classification_accuracy(type_prediction, tag_prediction, labels, tag_labels)

            type_accuracy += correct_type_num
            tag_accuracy += correct_tag_num

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        normalized_type_accuracy = type_accuracy / nb_eval_examples
        normalized_tag_accuracy = tag_accuracy / nb_eval_examples
        logger.info("Type accuracy: %.5f" % normalized_type_accuracy)
        results["normalized_type_accuracy"] = normalized_type_accuracy
        logger.info("Tag accuracy: %.5f" % normalized_tag_accuracy)
        results["normalized_tag_accuracy"] = normalized_tag_accuracy
        return results


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
    test_samples, test_num = data.read_split(params["data_path"], params['split'])
    if (logger):
        logger.info("Read %d test samples." % test_num)

    test_tensor_data = data.process_data(
        test_samples,
        tokenizer,
        params,
    )
    test_sampler = SequentialSampler(test_tensor_data)

    test_dataloader = DataLoader(
        test_tensor_data, sampler=test_sampler, batch_size=params['eval_batch_size']
    )

    # evaluate
    results = evaluate(
        model, test_dataloader, params, device=device, logger=logger,
    )
    
    out_file_path = os.path.join(params['output_path'], '%s_set_evaluation_results.json' %params['split'])
    with open(out_file_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, indent=4)


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
    parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    main(params)