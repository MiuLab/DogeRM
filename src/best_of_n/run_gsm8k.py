import argparse
import os
import sys
import json
import jsonlines
import torch
import transformers
import datasets
import logging
import pickle

from tqdm import tqdm
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torch.utils.data import DataLoader

from src.utils import set_seed, DEFAULT_CHAT_TEMPLATE, DEFAULT_SYSTEM_PROMPT
from bon_utils import BoNDataset, find_number, maybe_remove_comma, MAX_INT


logger = logging.getLogger(__name__)


def generate(args):
    lm_tokenizer = AutoTokenizer.from_pretrained(args.lm)
    if not lm_tokenizer.chat_template:
        lm_tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    # prepare_data
    system_prompt = "You are a math problem solver. Please think step by step and demonstrate your calculation steps. After your reasoning steps, you should generate the answer by following the format starting with 'The answer is'"
    original_ins, gsm8k_ins, gsm8k_answers = [], [], []
    with open(args.data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            dialogue = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["query"]}
            ]
            temp_instr = lm_tokenizer.apply_chat_template(dialogue, tokenize=False, add_generation_prompt=True) + " Let's think step by step."
            gsm8k_ins.append(temp_instr)
            original_ins.append(item['query'])
            temp_ans = item['response'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    # load language model
    llm = LLM(args.lm, dtype=torch.float16)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "[INST]", "<s>", "[/INST]"]
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=512, stop=stop_tokens)

    all_outputs = {}
    for n in tqdm(range(args.best_of), desc=f"generating response"):
        curr_outputs = llm.generate(gsm8k_ins, sampling_params)
        for i, output in enumerate(curr_outputs):
            generated_text = output.outputs[0].text
            if i not in all_outputs.keys():
                all_outputs[i] = defaultdict(list)
            
            all_outputs[i]['prompt'].append(original_ins[i])
            all_outputs[i]['answer'].append(generated_text)
            all_outputs[i]['ground_truth'].append(gsm8k_answers[i])
    
    # Store the output
    output_dir = os.path.dirname(args.lm_output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.lm_output_path, 'w') as f:
        print(json.dumps(all_outputs, indent=4), file=f)


def rerank(args):
    rm_builders = {
        "default": AutoModelForSequenceClassification.from_pretrained,
        "openbmb/Eurus-RM-7b": AutoModel.from_pretrained
    }
    build_function = rm_builders.get(args.rm, rm_builders["default"])
    reward_model = build_function(args.rm, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.rm)

    if not tokenizer.chat_template:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    if not reward_model.config.pad_token_id:
        reward_model.config.pad_token_id = tokenizer.pad_token_id
        reward_model.config.pad_token = tokenizer.pad_token
        
    with open(args.lm_output_path, 'r') as f:
        outputs = json.load(f)

    system_prompt = DEFAULT_SYSTEM_PROMPT if "llama" in args.rm else ""

    if args.debug:
        dataset = BoNDataset(outputs, tokenizer, args.best_of, size=10, system_prompt=system_prompt)
    else:
        dataset = BoNDataset(outputs, tokenizer, args.best_of, system_prompt=system_prompt)
        # dataset = BoNDataset(outputs, tokenizer, args.best_of)
    print(dataset[0])

    def collate_fn(batch):
        all_queries = [query for sublist in batch for query in sublist]
        tokenized_inputs = tokenizer(all_queries, padding=True, truncation=True, return_tensors="pt")

        return tokenized_inputs
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    reward_model = reward_model.to('cuda')
    reward_model.eval()
    results = []
    with torch.no_grad():
        for tokenized_inputs in tqdm(dataloader, desc="Select best reponse"):
            inputs = {k: v.to('cuda') for k, v in tokenized_inputs.items()}
            outputs = reward_model(**inputs)
            inputs = {k: v.to('cpu') for k, v in inputs.items()}
            del inputs
            logits = outputs.logits if "Eurus" not in args.rm else outputs
            logits = logits.view(-1, args.best_of)
            logits = logits.cpu().tolist()
            results += logits
            del outputs
    
    # return list of rewards
    return results


def pass_at_n(args, best_of=None):
    if not best_of:
        best_of = args.best_of
    
    with open(args.lm_output_path, 'r') as f:
        outputs = json.load(f)
    outputs = list(outputs.values())

    size = MAX_INT if not args.debug else 10

    answers = [
        [maybe_remove_comma(find_number(p)) for p in output['answer'][:best_of]] for output in outputs[:size]
    ]
    ground_truths = [output['ground_truth'][:best_of] for output in outputs[:size]]

    assert len(answers) == len(ground_truths)
    assert len(answers[-1]) == len(ground_truths[-1])

    result = [
        any((ans != '' and float(ans) == float(gd)) for ans, gd in zip(answer, ground_truth))
            for answer, ground_truth in zip(answers, ground_truths)
    ]

    return sum(result) / len(result)


def calculate_acc(args, results, best_of=None):
    if not best_of:
        best_of = args.best_of
    # Select best response for each question
    selected = []
    for result in results:
        select_idx = 0
        for idx in range(1, best_of):
            if result[idx] > result[select_idx]:
                select_idx = idx
        selected.append(select_idx)
    
    with open(args.lm_output_path, 'r') as f:
        outputs = json.load(f)
    outputs = list(outputs.values())
    if args.debug:
        outputs = outputs[:10]

    assert len(selected) == len(outputs), f'{len(selected)} != {len(outputs)}' 

    preds = [output['answer'][i] for output, i in zip(outputs, selected)]
    ground_truths = [output['ground_truth'][i] for output, i in zip(outputs, selected)]

    # answers = [extract_last_number(pred) for pred in preds]
    answers = [maybe_remove_comma(find_number(pred)) for pred in preds]
    correct_cnt = 0
    for answer, ground_truth in zip(answers, ground_truths):
        if answer != '' and float(answer) == float(ground_truth):
            correct_cnt += 1
    acc = correct_cnt / len(answers)
    
    return acc


def main():
    parser = argparse.ArgumentParser()
    # Models
    parser.add_argument('--lm', type=str, default="/work/hank0316/LLaMA-Factory/saves/LLaMA2-7B/full/sft-warmup/")
    parser.add_argument('--rm', type=str, default="/work/hank0316/reward-model-merge/models/llama2-7b-alpaca-preference-2epoch")
    parser.add_argument('--base_model', type=str, default="llama2")

    # Task & data
    parser.add_argument('--data_path', type=str, default="data/GSM8K_test.jsonl", help="Data path for llm generation")

    parser.add_argument('--best_of', type=int, default=2, help="Number of candidate to be selected")
    parser.add_argument('--lm_output_path', type=str, default='../../bon_output/model_output/gsm8k-llama2-chat.json')
    parser.add_argument('--rm_output_path', type=str, default='../../bon_output/rm_output/rerank_output.pkl')
    parser.add_argument('--log_file', type=str, default="../../bon_output/log_files/log.txt")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, choices=['generate', 'rerank', 'cal_acc'], default='generate')
    parser.add_argument('--debug', action='store_true', help='If set, only 10 rows will be used')

    args = parser.parse_args()

    # fix random seed for reproducibility
    set_seed(args.seed)

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.set_verbosity_info()

    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if args.mode == 'generate':
        generate(args)
    elif args.mode == 'rerank':
        results = rerank(args)
        with open(args.log_file, 'a+') as log_file:
            print(args, file=log_file)
            curr = 1
            while curr <= args.best_of:
                # log from 1, 2, 4, ..., args.best_of
                pass_at_n_acc = pass_at_n(args, curr)
                if curr > 1:
                    rerank_acc = calculate_acc(args, results, curr)
                    print(f"Best-of-{curr} result: pass@{curr} = {pass_at_n_acc}, rerank_acc = {rerank_acc}", file=log_file)
                else:
                    print(f"Best-of-{curr} result: pass@{curr} = {pass_at_n_acc}", file=log_file)
                
                curr = curr << 1
            print('='*20, file=log_file)

        # save result
        output_dir = os.path.dirname(args.rm_output_path)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        with open(args.rm_output_path, 'wb') as result_f:
            pickle.dump(results, result_f)
    
    elif args.mode == 'cal_acc':
        with open(args.rm_output_path, 'rb') as result_f:
            results = pickle.load(result_f)

        with open(args.log_file, 'a+') as log_file:
            print(args, file=log_file)

            pass_at_n_acc = pass_at_n(args)
            if args.best_of > 1:
                rerank_acc = calculate_acc(args, results, args.best_of)
                print(f"Best-of-{args.best_of} result: pass@{args.best_of} = {pass_at_n_acc}, rerank_acc = {rerank_acc}", file=log_file)
            else:
                print(f"Best-of-{args.best_of} result: pass@{args.best_of} = {pass_at_n_acc}", file=log_file)
            
            print('='*20, file=log_file)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
