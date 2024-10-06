#!/bin/bash

: '
This script does the folowing two things:
    1. Merge reward model with language model
    2. Rank model response on GSM8k

Execution:
bash merge_gsm8k_bon.sh \
    $reward_model \
    $language_model \
    $output_root \
    $log_file \
    $score_file_prefix \
    $range_split
'

reward_model="$1"
language_model="$2"
merged_model_path="$3"
output_root="$4"        # root dir for best-of-n output
lm_output_name="$5"     # filename only
log_file_name="$6"      # filename only
score_file_prefix="$7"

echo "Merge $reward_model with $language_model"

export PYTHONPATH="$(dirname $(pwd)):$PYTHONPATH"

# run the unmerged RM
rm_output_dir=$output_root/rm_output
log_file_dir=$output_root/log_file

mkdir -p $rm_output_dir
mkdir -p $log_file_dir

log_file=$log_file_dir/$log_file_name

for rm_ratio in $(seq 1.0 -0.05 0); do
    ratio=$(echo "1 - $rm_ratio" | bc);
    lm_ratio=$(printf "%.2f" $ratio);
    echo "seq_ratio: $rm_ratio, lm_ratio: $lm_ratio"

    # Merge model
    if [[ $rm_ratio != 1.00 ]]; then
        python ../src/merge_linear.py \
            --seq_model $reward_model \
            --seq_weight $rm_ratio \
            --lm_model $language_model \
            --lm_weight $lm_ratio \
            --output_path $merged_model_path
        resulting_rm=$merged_model_path
    else
        resulting_rm=$reward_model
    fi
    
    scoring_file="$score_file_prefix-seq-$rm_ratio-lm-$lm_ratio.pkl"
    echo "generate output to $rm_output_dir/$scoring_file"

    # Generating scoring of each candidate and store the scores
    python ../src/best_of_n/run_gsm8k.py \
        --rm $resulting_rm \
        --lm_output_path $output_root/model_output/$lm_output_name \
        --rm_output_path "$rm_output_dir/$scoring_file" \
        --best_of 16 \
        --batch_size 1 \
        --mode rerank \
        --log_file $log_file
done
