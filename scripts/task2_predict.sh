CUDA_VISIBLE_DEVICES=4 python ./src/task2/predict.py \
    --data_path ./data/raw/task2_new \
    --output_path ./data/model/task2_new \
    --load_model_path ./data/model/task2_new/checkpoint.bin \
    --base_model hfl/chinese-bert-wwm-ext \
    --seq_max_length 256 \
    --split dev \
    --seed 42 \
    --cuda 