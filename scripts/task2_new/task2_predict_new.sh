CUDA_VISIBLE_DEVICES=0 python ./src/task2_new/predict.py \
    --data_path ./data/input/task2 \
    --output_path ./data/model/task2_new \
    --load_model_path ./data/model/task2_new \
    --base_model hfl/chinese-bert-wwm-ext \
    --seq_max_length 256 \
    --split test \
    --seed 42 \
    --cuda 