import os
import argparse
import torch
import json
import random
import time
import numpy as np

from tqdm import tqdm, trange

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

import data


def main(params):
    tokenizer = AutoTokenizer.from_pretrained(params['base_model'], model_max_length=512)
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    train_raw = data.read_split(params['data_path'], 'train')
    train_dataset = data.process_data(
        train_raw,
        tokenizer,
        params,
        test_mode=False,
    )

    # print(train_dataset)
    # input()

    args = Seq2SeqTrainingArguments(
        params['output_path'],
        evaluation_strategy="steps",
        eval_steps=params['eval_interval'],
        logging_strategy="steps",
        logging_steps=params['print_interval'],
        save_strategy="steps",
        save_steps=params['save_interval'],
        learning_rate=params['learning_rate'],
        per_device_train_batch_size=params['train_batch_size'],
        per_device_eval_batch_size=params['eval_batch_size'],
        weight_decay=0.01,
        save_total_limit=4,
        num_train_epochs=params['epoch'],
        predict_with_generate=True,
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="none"
    )

    def init_model():
        model = AutoModelForSeq2SeqLM.from_pretrained(params['base_model'])
        if (params['cuda']):
            model = model.to('cuda')
        return model

    trainer = Seq2SeqTrainer(
        model_init=init_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--data_path', type=str, default='./data/input/task3')
    parser.add_argument('--output_path', type=str, default='./data/model/task3_generative')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--base_model', type=str, default="google/mt5-base")
    
    # model arguments
    parser.add_argument('--seq_max_length', type=int, default=256)
    parser.add_argument('--cuda', action='store_true')

    # training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    main(params)