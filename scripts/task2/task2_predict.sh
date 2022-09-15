CUDA_VISIBLE_DEVICES=4 python ./src/task2/predict.py \
    --data_path ./data/input/task2 \
    --output_path ./data/model/task2 \
    --load_model_path ./data/model/task2/checkpoint.bin \
    --base_model hfl/chinese-bert-wwm-ext \
    --seq_max_length 256 \
    --split test \
    --seed 42 \
    --cuda 