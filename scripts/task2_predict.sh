CUDA_VISIBLE_DEVICES=4 python ./src/task2/predict.py \
    --data_path ./data/raw/task2 \
    --output_path ./data/model/task2 \
    --load_model_path ./data/model/task2/checkpoint.bin \
    --base_model bert-base-chinese \
    --seq_max_length 256 \
    --split dev \
    --seed 42 \
    --cuda 