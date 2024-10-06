import os
import re
import json
import pickle
import argparse

from bon_utils import argmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_root',
        type=str, 
        default="../../bon_output/llama2/humaneval",
        help="root dir for lm_output_name & rm_output_name"
    )
    parser.add_argument('--lm_output_name', type=str, required=True)
    parser.add_argument('--rm_output_name', type=str, required=True)
    parser.add_argument('--best_of', type=int, default=2)
    parser.add_argument('--task', type=str, choices=['humaneval', 'mbpp', 'math'], default='humaneval')
    parser.add_argument('--result_path', type=str, default="")
    parser.add_argument('--debug', action="store_true", help="Use 10 data if set to True")

    args = parser.parse_args()
    print(args)

    lm_output_path = os.path.join(args.output_root, 'model_output', args.lm_output_name)
    rm_output_path = os.path.join(args.output_root, 'rm_output', args.rm_output_name)

    with open(lm_output_path, 'r') as lm_f:
        lm_output = json.load(lm_f)
        lm_output = list(lm_output.values())
    
    with open(rm_output_path, 'rb') as rm_f:
        rm_output = pickle.load(rm_f)

    if args.debug:
        lm_output = lm_output[:10]
        rm_output = rm_output[:10]

    if args.task == 'humaneval':
        stop_words = ["\n```", "\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\nassert", "<\s>"]
    elif args.task == 'mbpp':
        stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```", "<\s>"]

    selected = []
    for lm_out, rm_out in zip(lm_output, rm_output):
        rm_out = rm_out[:args.best_of]
        selected_idx = argmax(rm_out)

        if args.task == 'humaneval':
            for w in stop_words:
                if w in lm_out['outputs'][selected_idx]:
                    end_idx = lm_out['outputs'][selected_idx].find(w)
                    lm_out['outputs'][selected_idx] = lm_out['outputs'][selected_idx][:end_idx]
            
            code_block = lm_out['prompt'][selected_idx] + lm_out['outputs'][selected_idx]
            selected.append([code_block.strip()])
        elif args.task == 'mbpp':
            # search start of the code block
            code_block = lm_out['outputs'][selected_idx]
            for start_pattern in ['```python', '```']:
                if start_pattern in code_block:
                    start_idx = code_block.find(start_pattern) + len(start_pattern)
                    code_block = code_block[start_idx:].strip()
                    break
            
            code_block = lm_out['outputs'][selected_idx][start_idx:]
            for w in stop_words:
                if w in code_block:
                    end_idx = code_block.find(w)
                    code_block = code_block[:end_idx]
            
            selected.append([code_block.strip()])
        else:
            res = {
                "outputs": lm_out['outputs'][selected_idx],
                "expected_answer": lm_out['expected_answer'][selected_idx]
            }
            selected.append(res)

    result_dir = os.path.dirname(args.result_path)
    if result_dir != '' and not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    with open(args.result_path, 'w+') as result_f:
        json.dump(selected, result_f)

    print(f'result is stored to {args.result_path}...')
    

if __name__ == '__main__':
    main()