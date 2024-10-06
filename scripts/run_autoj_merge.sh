#!/bin/bash

reward_model="$1"
language_model="$2"
merged_model_path="$3"
autoj_data_path="$4"

echo Running unmerged RM $reward_model
python ../src/inference_autoj.py \
    --model $reward_model \
    --data_path $autoj_data_path

for rm_ratio in $(seq 0.95 -0.05 0.0); do
    ratio=$(echo "1 - $rm_ratio" | bc);
    lm_ratio=$(printf "%.2f" $ratio);
    echo "seq_ratio: $rm_ratio, lm_ratio: $lm_ratio"
    # echo "seq_ratio: $rm_ratio, lm_ratio: $lm_ratio" >> $log_file

    # Merge model
    python ../src/merge_linear.py \
        --seq_model $reward_model \
        --seq_weight $rm_ratio \
        --lm_model $language_model \
        --lm_weight $lm_ratio \
        --output_path $merged_model_path

    # Run inference on AutoJ
    python ../src/inference_autoj.py \
        --model $merged_model_path \
        --data_path $autoj_data_path

done