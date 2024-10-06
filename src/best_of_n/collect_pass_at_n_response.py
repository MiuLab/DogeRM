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
    parser.add_argument('--best_of', type=int, default=2)
    parser.add_argument('--task', type=str, choices=['humaneval', 'mbpp'], default='humaneval')
    parser.add_argument('--result_path', type=str, default="")
    parser.add_argument('--debug', action="store_true", help="Use 10 data if set to True")

    args = parser.parse_args()
    print(args)

    lm_output_path = os.path.join(args.output_root, 'model_output', args.lm_output_name)
    
    with open(lm_output_path, 'r') as lm_f:
        lm_output = json.load(lm_f)
        lm_output = list(lm_output.values())
    
    if args.debug:
        lm_output = lm_output[:5]

    if args.task == 'humaneval':
        stop_words = ["\n```", "\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\nassert"]
    elif args.task == 'mbpp':
        stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"]
    else:
        raise NotImplementedError
    stop_words.append("</s>")

    notfound_cnt = 0
    results = []
    for lm_out in lm_output:
        result = [] # response of one question
        if args.task == 'humaneval':
            for i in range(args.best_of):
                # truncation
                for w in stop_words:
                    if w in lm_out['outputs'][i]:
                        end_idx = lm_out['outputs'][i].find(w)
                        lm_out['outputs'][i] = lm_out['outputs'][i][:end_idx]
                
                code_block = lm_out['prompt'][i] + lm_out['outputs'][i]
                result.append(code_block)
        elif args.task == 'mbpp':
            for i in range(args.best_of):
                # find the start of code block
                code_block = lm_out['outputs'][i]
                print('\n-----')
                print(f'before parsing, code:\n{code_block}')
                for start_pattern in ['```python', '```']:
                    if start_pattern in code_block:
                        start_idx = code_block.find(start_pattern) + len(start_pattern)
                        code_block = code_block[start_idx:].strip()
                        break
                
                for w in stop_words:
                    if w in code_block:
                        end_idx = code_block.find(w)
                        code_block = code_block[:end_idx]
                print(f'after parsing, code:\n{code_block}')
                print('-----\n')
                result.append(code_block)
        else:
            result.append(lm_out['outputs'][i])
        
        results.append(result)
    
    result_dir = os.path.dirname(args.result_path)
    if result_dir != '' and not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    with open(args.result_path, 'w+') as result_f:
        json.dump(results, result_f)

    # print(f'parse failed: {notfound_cnt}')
    print(f'results are stored to {args.result_path}...')
    

if __name__ == '__main__':
    main()