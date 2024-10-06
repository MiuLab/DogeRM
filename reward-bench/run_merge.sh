#!/bin/sh

seq_model="$1"
lm_model="$2"
output_path="$3"
proj_name="$4"
run_prefix="$5"

echo merge $seq_model with $lm_model

for seq_ratio in $(seq 1 -0.05 0); do
    ratio=$(echo "1 - $seq_ratio" | bc);
    lm_ratio=$(printf "%.2f" $ratio);
    echo "lm_ratio: $lm_ratio";
    if [[ $seq_ratio != 1.00 ]]; then
        python ../reward-model-merge/src/merge_linear.py --seq_model $seq_model --seq_weight $seq_ratio --lm_model $lm_model --lm_weight $lm_ratio --output_path $merged_model_output_path;
        reward_model=$merged_model_output_path
    else
        reward_model=$seq_model
    fi
    python scripts/run_rm.py --batch_size 4 --model $reward_model --do_not_save --seq_weight $seq_ratio --lm_weight $lm_ratio --to_wandb --proj_name $proj_name --run_name $run_prefix-seq-$seq_ratio-lm-$lm_ratio;
done