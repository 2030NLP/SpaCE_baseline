import os
import argparse
import torch
import json
import random
import time
import numpy as np

from tqdm import tqdm, trange

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from model import Task1Model
import data
import utils


def load_model(params):
    return Task1Model(params)


def evaluate(
    model, eval_dataloader, params, device, logger,
):
    with torch.no_grad():
        model.eval()
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

        results = {}

        eval_accuracy = 0.0
        nb_eval_examples = 0
        nb_eval_steps = 0
    
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            input_ids, token_type_ids, attention_mask, labels = batch
                
            logits, loss = model(
                input_ids, 
                token_type_ids, 
                attention_mask,
                labels,
            )

            logits = logits.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            tmp_eval_accuracy, _ = utils.binary_accuracy(logits, labels)

            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
        logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
        results["normalized_accuracy"] = normalized_eval_accuracy
        return results


def main(params):
    model_output_path = params['output_path']
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    logger = utils.setup_logger('SpaCET1', params['output_path'])

    # Init model
    model = Task1Model(params)
    tokenizer = model.tokenizer
    device = model.device

    model = model.to(model.device)

    if params['gradient_accumulation_steps'] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params['gradient_accumulation_steps']
            )
        )

    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load train data
    train_samples, train_num = data.read_split(params["data_path"], "train")
    if (logger):
        logger.info("Read %d train samples." % train_num)

    train_tensor_data = data.process_data(
        train_samples,
        tokenizer,
        params,
    )

    if params["shuffle"]:
        train_sampler = RandomSampler(train_tensor_data)
    else:
        train_sampler = SequentialSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
    )

    # Load eval data
    valid_samples, valid_num = data.read_split(params["data_path"], "dev")
    if (logger):
        logger.info("Read %d valid samples." % valid_num)

    valid_tensor_data = data.process_data(
        valid_samples,
        tokenizer,
        params,
    )

    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    # evaluate before training
    results = evaluate(
        model, valid_dataloader, params, device=device, logger=logger,
    )

    time_start = time.time()

    param_path = os.path.join(model_output_path, 'config.json')
    with open(param_path, 'w', encoding='utf-8') as fout:
        fout.write(json.dumps(params))

    if (logger):
        logger.info("Start training")

    optimizer = utils.get_optimizer(model, params)
    scheduler = utils.get_scheduler(params, optimizer, len(train_tensor_data), logger)

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["epoch"]
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        model.train()
        tr_loss = 0
        results = None
        iter_ = tqdm(train_dataloader, desc="Batch")

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            input_ids, token_type_ids, attention_mask, labels = batch
                
            logits, loss = model(
                input_ids, 
                token_type_ids, 
                attention_mask,
                labels,
            )

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                if (logger):
                    logger.info(
                        "Step %d - epoch %d average loss: %.4f;" %(
                            step,
                            epoch_idx,
                            tr_loss / (params["print_interval"] * grad_acc_steps),
                        )
                    )
                tr_loss = 0
                # print(logits)
                # print(labels)

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        if (logger):
            logger.info("***** Saving fine - tuned model *****")

        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_%d" %(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            model, valid_dataloader, params, device=device, logger=logger,
        )
        with open(output_eval_file, 'w', encoding='utf-8') as fout:
            fout.write(json.dumps(results, indent=4))

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        if (logger):
            logger.info("\n")

    execution_time = (time.time() - time_start) / 60

    _path = os.path.join(model_output_path, "training_time.txt")
    with open(_path, 'w', encoding='utf-8') as fout:
        fout.write('The training took %f minutes\n' %execution_time)
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["load_model_path"] = os.path.join(
        model_output_path, 
        "epoch_%d" %(best_epoch_idx),
        'checkpoint.bin',
    )

    model = load_model(params)
    utils.save_model(model, tokenizer, model_output_path)

    if params["final_evaluate"]:
        params["load_model_path"] = model_output_path
        evaluate(model, valid_dataloader, params, device=device, logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--data_path', type=str, default='./data/raw/task1')
    parser.add_argument('--output_path', type=str, default='./data/model/task1')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--base_model', type=str, default='hfl/chinese-bert-wwm-ext')
    
    # model arguments
    parser.add_argument('--seq_max_length', type=int, default=256)
    parser.add_argument('--cuda', action='store_true')

    # training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--print_interval', type=int, default=5)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--final_evaluate', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    main(params)