import os
import re
import sys
import json
import torch
import pickle
import logging
import argparse
import datasets
import transformers

from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from src.utils import set_seed, DEFAULT_CHAT_TEMPLATE, DEFAULT_SYSTEM_PROMPT
from bon_utils import MbppProcessor, MbppDataset, argmax


logger = logging.getLogger(__name__)


def generate(args):
    tokenizer = AutoTokenizer.from_pretrained(args.lm)
    if not tokenizer.chat_template:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset(args.data_path, split=args.data_split)
    
    # get function name of each task to fill it into input prompt
    if 'mistral' not in args.lm.lower():
        system_prompt = f"Write Python code to solve the task."
    else:
        system_prompt = ""
    dataset = dataset.map(
        MbppProcessor.prepare_example, 
        fn_kwargs={
            "tokenizer": tokenizer,
            "system_prompt": system_prompt
        }
    )
    queries = [data['query'] for data in dataset]
    if args.debug:
        queries = queries[:10]
    
    for i in range(5):
        print(queries[i])
        print('---')
        # exit(0)
    llm = LLM(args.lm, dtype=torch.float16)
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=512,
        stop=[tokenizer.eos_token]
    )

    all_outputs = {}
    for n in tqdm(range(args.best_of), desc=f"generating response"):
        curr_outputs = llm.generate(queries, sampling_params)
        for i, output in enumerate(curr_outputs):
            generated_text = output.outputs[0].text
            if i not in all_outputs.keys():
                all_outputs[i] = defaultdict(list)
            for k, v in dataset[i].items():
                all_outputs[i][k].append(v)
            all_outputs[i]['outputs'].append(generated_text)

    # Store the output
    output_dir = os.path.dirname(args.lm_output_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
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
    
    with open(args.lm_output_path, 'r') as f:
        outputs = json.load(f)

    system_prompt = DEFAULT_SYSTEM_PROMPT if 'llama2' in args.rm else ''
    if args.debug:
        dataset = MbppDataset(outputs, tokenizer, args.best_of, size=10, system_prompt=system_prompt, output_key='outputs')
    else:
        dataset = MbppDataset(outputs, tokenizer, args.best_of, system_prompt=system_prompt, output_key='outputs')
        # dataset = MbppDataset(outputs, tokenizer, args.best_of)
    for i in range(5):
        print(dataset[i][-1])
    
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


def main():
    parser = argparse.ArgumentParser()
    # Models
    parser.add_argument('--lm', type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--rm', type=str, default="/work/hank0316/reward-model-merge/models/llama2-7b-alpaca-preference-2epoch")

    # Task & data
    parser.add_argument('--data_path', type=str, default="mbpp", help="Data path for llm generation")
    # parser.add_argument('--language', type=str, default='python', choices=['js', 'java', 'python', 'cpp', 'go', 'rust'])
    parser.add_argument('--data_split', type=str, default='test')

    parser.add_argument('--best_of', type=int, default=2, help="Number of candidate to be selected")
    parser.add_argument('--lm_output_path', type=str, default='../../bon_output/model_output/llama2-chat.json')
    parser.add_argument('--rm_output_path', type=str, default='../../bon_output/rm_output/rerank_output.pkl')
    parser.add_argument('--log_file', type=str, default="../../bon_output/log_files/log.txt")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, choices=['generate', 'rerank'], default='generate')
    parser.add_argument('--debug', action='store_true', help='If set, only 10 rows will be used')

    args = parser.parse_args()
    print(args)
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
        # save result
        output_dir = os.path.dirname(args.rm_output_path)
        if output_dir != "" and not os.path.exists(output_dir):
            os.mkdir(output_dir)

        with open(args.rm_output_path, 'wb') as result_f:
            pickle.dump(results, result_f)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
