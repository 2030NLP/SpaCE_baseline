CUDA_VISIBLE_DEVICES=0 python ./src/task3_extractive/predict.py \
    --data_path ./data/input/task3 \
    --output_path ./data/model/task3_extractive \
    --load_model_path ./data/model \
    --base_model hfl/chinese-bert-wwm-ext \
    --seq_max_length 256 \
    --split test \
    --eval_batch_size 32 \
    --seed 42 \
    --cuda 