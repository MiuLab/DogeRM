import json
import pandas as pd
import argparse
import torch

from src.utils import DEFAULT_SYSTEM_PROMPT
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')

    args = parser.parse_args()

    if 'jsonl' in args.data_path:
        dataset = pd.read_json(args.data_path, lines=True)
    elif 'json' in args.data_path:
        with open(args.data_path, 'r') as f:
            dataset = json.load(f)
    else:
        dataset = load_dataset(args.data_path, split=args.split)

    def categorize_scenario(scenario):
        if scenario in ['math_reasoning', 'solving_exam_question_with_math']:
            return 'Math'
        elif scenario in ['code_simplification', 'code_generation', 'explaining_code', 'code_to_code_translation', 'code_correction_rewriting']:
            return 'Code'
        else:
            return 'Other'

    dataset = dataset[dataset['label'] != 2]
    dataset['category'] = dataset['scenario'].apply(categorize_scenario)
    # dataset = dataset[dataset['category'].isin(['Code', 'Math'])]
    print(dataset)
    count_per_category = dataset.groupby('category').size().reset_index(name='count')
    print(count_per_category)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, torch_dtype=torch.float16) if "Eurus" not in args.model else AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    system_prompt = DEFAULT_SYSTEM_PROMPT if 'llama2' in args.model else ''
    # prepare dataset
    def prepare_data(example):
        if example['label'] == 0:
            chosen = example['response 1']
            rejected = example['response 2']
        else:
            chosen = example['response 2']
            rejected = example['response 1']
        
        # add system prompt if needed
        if system_prompt:
            chosen_full = [{'role': 'system', 'content': system_prompt}]
            rejected_full = [{'role': 'system', 'content': system_prompt}]
        else:
            chosen_full = []
            rejected_full = []

        chosen_full += [
            {'role': 'user', 'content': example['prompt']},
            {'role': 'assistant', 'content': chosen}
        ]
        rejected_full += [
            {'role': 'user', 'content': example['prompt']},
            {'role': 'assistant', 'content': rejected}
        ]
        example['chosen'] = tokenizer.apply_chat_template(chosen_full, tokenize=False)
        example['rejected'] = tokenizer.apply_chat_template(rejected_full, tokenize=False)
        return example
    
    dataset = dataset.apply(prepare_data, axis=1)

    chosen = dataset['chosen'].tolist()
    rejected = dataset['rejected'].tolist()

    preds = []
    correct_cnt = 0
    model = model.to('cuda')
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(chosen))):
            inputs = tokenizer([chosen[i], rejected[i]], return_tensors="pt", padding=True, max_length=2048, truncation=True)
            input_ids = inputs['input_ids'].to('cuda')
            attention_mask = inputs['attention_mask'].to('cuda')
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            if "Eurus" not in args.model:
                output = output.logits
            rewards = output

            correct_cnt += (rewards[0] > rewards[1])
            preds.append(1 if (rewards[0] > rewards[1]) else 0)
    
    dataset['pred'] = preds
    accuracy_per_category = dataset.groupby('category')['pred'].mean().reset_index()
    accuracy_per_category.columns = ['category', 'accuracy']
    print(accuracy_per_category)
    print(f"Overall Accuracy: {correct_cnt/len(chosen)}")


if __name__ == '__main__':
    main()
