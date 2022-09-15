PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=4 python ./src/task3_generative/predict.py \
    --data_path ./data/input/task3 \
    --output_path ./data/model/task3_generative \
    --load_model_path ./data/model/task3_generative/checkpoint-1600 \
    --base_model google/mt5-base \
    --seq_max_length 512 \
    --eval_batch_size 8 \
    --cuda 