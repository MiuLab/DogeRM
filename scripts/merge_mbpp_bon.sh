#!/bin/bash
set -x 
: '
This script does the following two things:
    1. Merge reward model with language model
    2. Use the merged reward model to rerank model responses on HumanEval

Execution:
bash merge_mbpp_bon.sh \
    $reward_model \
    $language_model \
    $merged_model_path \
    $output_root \          # the root directory of (1) model_output, (2) rm_output, and (3) log_file
    $lm_output_name \       # the file name of model response, this file is located in $output_root/model_output/
    $log_file_name \        # the file name of log file, this file will be stored at $output_root/log_file/
    $score_file_prefix      # the file_name prefix of RM reranking output
    $path_to_bigcode_eval   # path to bigcode-evaluation-harness
'

reward_model="$1"
language_model="$2"
merged_model_path="$3"
output_root="$4" # root dir for best-of-n output
lm_output_name="$5" # filename only
log_file_name="$6" # filename only
score_file_prefix="$7" 
path_to_bigcode_eval="$8"

echo Merge $reward_model with $language_model
echo output_root: $output_root

lm_output_dir="$output_root/model_output"
rm_output_dir="$output_root/rm_output"
selected_output_dir="$output_root/selected_output"
mkdir -p $rm_output_dir
mkdir -p $selected_output_dir

log_file="$output_root/log_file/$log_file_name"
echo "Write log to $log_file..."
echo "" > $log_file

export PYTHONPATH="$(dirname $(pwd)):$PYTHONPATH"

lm_output_path="$lm_output_dir/$lm_output_name"
python ../src/best_of_n/run_mbpp.py \
        --lm meta-llama/Llama-2-7b-chat-hf \
        --lm_output_path "$lm_output_path" \
        --best_of 16 \
        --batch_size 1 \
        --mode generate \
        --log_file $log_file

for rm_ratio in $(seq 1.0 -0.05 0); do
    ratio=$(echo "1 - $rm_ratio" | bc);
    lm_ratio=$(printf "%.2f" $ratio);
    echo "seq_ratio: $rm_ratio, lm_ratio: $lm_ratio"
    echo "seq_ratio: $rm_ratio, lm_ratio: $lm_ratio" >> $log_file

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
        echo "use the unmerged one"
        resulting_rm=$reward_model
    fi

    scoring_file="$score_file_prefix-seq-$rm_ratio-lm-$lm_ratio.pkl"
    rm_output_path="$rm_output_dir/$scoring_file"

    echo "generate output to $rm_output_path"

    # Generating scoring of each candidate and store the scores
    python ../src/best_of_n/run_mbpp.py \
        --rm $resulting_rm \
        --lm_output_path "$lm_output_path" \
        --rm_output_path "$rm_output_path" \
        --best_of 16 \
        --batch_size 1 \
        --mode rerank \
        --log_file $log_file

    # extract the response selected by RM
    for best_of in 2 4 8 16; do
        rm_selected_output_path=$selected_output_dir/$best_of/$score_file_prefix-seq-$rm_ratio-lm-$lm_ratio.json
        python ../src/best_of_n/get_selected_response.py \
            --output_root $output_root \
            --lm_output_name $lm_output_name \
            --rm_output_name $scoring_file \
            --best_of $best_of \
            --task mbpp \
            --result_path $rm_selected_output_path

        execution_result_dir=$output_root/execution_result/$best_of
        mkdir -p $execution_result_dir
        accelerate launch $path_to_bigcode_eval/main.py \
            --tasks mbpp \
            --allow_code_execution \
            --load_generations_path $rm_selected_output_path \
            --metric_output_path $execution_result_dir/$score_file_prefix-seq-$rm_ratio-lm-$lm_ratio.json
    done
done
