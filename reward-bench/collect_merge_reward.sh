#!/bin/bash

#!/bin/sh

export HF_HOME="/work/hank0316/.cache"
export TRANSFORMERS_CACHE="/work/hank0316/.cache";
export HF_DATASETS_CACHE="/work/hank0316/.cache";

seq_model="$1"
lm_model="$2"
output_path="$3"
range_split="$4"

echo merge $seq_model with $lm_model

if [ $range_split -eq 1 ]; then
    range=$(seq 0 0.05 0.45)
elif [ $range_split -eq 2 ]; then
    range=$(seq 0.5 0.05 0.95)
elif [ $range_split -eq 3 ]; then
    range=$(seq 0 0.05 0.95)
elif [ $range_split -eq 4 ]; then
    # test for one run
    range=$(seq 0.5 0.05 0.5)
else
    echo "invalid range_split argument, please specified 1(first-half), 2(second-half), or 3(all)"
    exit -1
fi

for seq_ratio in $range; do
    ratio=$(echo "1 - $seq_ratio" | bc);
    lm_ratio=$(printf "%.2f" $ratio);
    echo "lm_ratio: $lm_ratio";
    python ../reward-model-merge/src/merge_linear.py --seq_model $seq_model --seq_weight $seq_ratio --lm_model $lm_model --lm_weight $lm_ratio --output_path $output_path;
    python scripts/run_rm.py --batch_size 8 --model $output_path --do_not_save --seq_weight $seq_ratio --lm_weight $lm_ratio  --collect_reward;
    mv results/eval-set-scores/$output_path.json results/hep-all-score/seq-$seq_ratio-lm-$lm_ratio.json
done