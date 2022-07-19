# CUDA_VISIBLE_DEVICES=4 python ./src/task2/train.py \
#     --data_path ./data/raw/task2 \
#     --output_path ./data/model/task2 \
#     --base_model hfl/chinese-bert-wwm-ext \
#     --seq_max_length 256 \
#     --learning_rate 1e-5 \
#     --epoch 4 \
#     --train_batch_size 4 \
#     --eval_batch_size 8 \
#     --print_interval 20 \
#     --eval_interval 100 \
#     --shuffle \
#     --final_evaluate \
#     --seed 42 \
#     --cuda 

CUDA_VISIBLE_DEVICES=4 python ./src/task2/train.py \
    --data_path ./data/raw/task2 \
    --output_path ./data/model/task2 \
    --base_model bert-base-chinese \
    --seq_max_length 256 \
    --learning_rate 1e-5 \
    --epoch 4 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --print_interval 20 \
    --eval_interval 100 \
    --shuffle \
    --final_evaluate \
    --seed 42 \
    --cuda 