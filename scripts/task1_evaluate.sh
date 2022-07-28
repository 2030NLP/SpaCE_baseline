# CUDA_VISIBLE_DEVICES=4 python ./src/task1/evaluate.py \
#     --data_path ./data/raw/task1 \
#     --output_path ./data/model/task1 \
#     --load_model_path ./data/model/task1/checkpoint.bin \
#     --base_model hfl/chinese-bert-wwm-ext \
#     --seq_max_length 256 \
#     --split dev \
#     --seed 42 \
#     --cuda 

CUDA_VISIBLE_DEVICES=4 python ./src/task1/evaluate.py \
    --data_path ./data/raw/task1_new \
    --output_path ./data/model/task1_new \
    --load_model_path ./data/model/task1_new/checkpoint.bin \
    --base_model hfl/chinese-bert-wwm-ext \
    --seq_max_length 256 \
    --split dev \
    --seed 42 \
    --cuda 