import os
import argparse
import numpy as np

from tqdm import tqdm, trange

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
)

import data


def main(params):
    tokenizer = AutoTokenizer.from_pretrained(params['load_model_path'])
    model = AutoModelForSeq2SeqLM.from_pretrained(params['load_model_path'])
    if (params['cuda']):
        model.to('cuda')

    max_input_length = params['seq_max_length']

    test_raw = data.read_split(params['data_path'], 'test')
    test_dataset = data.process_data(
        test_raw,
        tokenizer,
        params,
        test_mode=True,
    )

    qids = test_dataset['qids']
    texts = test_dataset['text']
    input_ids = test_dataset['input_ids']
    num_examples = len(qids)
    all_decoded_results = []

    for i in range(num_examples):
        qid = qids[i]
        batch = {
            'input_ids': input_ids[i].unsqueeze(0).to(model.device),
        }

        output = model.generate(
            **batch, 
            num_beams=8, 
            do_sample=True, 
            min_length=2, 
            max_length=params['seq_max_length'],
        )
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        print(texts[i])
        print(decoded_output)
        input()
        # predicted_outputs = data.deserialize(decoded_output)

        # print(predicted_outputs[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--data_path', type=str, default='./data/input/task3')
    parser.add_argument('--output_path', type=str, default='./data/model/task3_generative')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--base_model', type=str, default="google/mt5-base")
    
    # model arguments
    parser.add_argument('--seq_max_length', type=int, default=512)
    parser.add_argument('--cuda', action='store_true')

    # training arguments
    parser.add_argument('--eval_batch_size', type=int, default=8)

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    main(params)