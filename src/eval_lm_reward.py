import torch
import argparse
import json
import random as rd

from collections import defaultdict
from datasets import load_dataset
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = 'Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given assistant A’s answer, and assistant B’s answer. Your job is to evaluate which assistant’s answer is better. You should independently solve the user question step-by-step first. Then compare both assistants’ answers with your answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by **strictly** following this format: "Final verdict: [[A]]" if assistant A is better, "Final verdict: [[B]]" if assistant B is better, and "Final verdict: [[C]]" for a tie.'
INSTRUCTION = '[User Question]\n{question}\n\n[The Start of Assistant A’s Answer]\n{answer_a}\n[The End of Assistant A’s Answer]\n\n[The Start of Assistant B’s Answer]\n{answer_b}\n[The End of Assistant B’s Answer]'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--task', type=str, default='math', choices=['math', 'code'])
    parser.add_argument('--subset', type=str, default="cpp", choices=['cpp', 'java', 'python', 'js', 'go', 'rust'])
    args = parser.parse_args()

    llm = LLM(model=args.model, dtype=torch.float16)
    tokenizer = llm.get_tokenizer()
    # tokenizer.model_max_length=4096
    dataset = load_dataset('allenai/reward-bench', split="train")

    if args.task == 'math':
        dataset = dataset.filter(lambda example: example['subset'] == 'math-prm')
    else:
        dataset = dataset.filter(lambda example: example['subset'] == f'hep-{args.subset}')
    
    def process_example(example):
        question = example['prompt']
        p = rd.uniform(0, 1)    # Random shuffle to avoid position bias
        if p <= 0.5:
            answer_a = example['chosen']
            answer_b = example['rejected']
            label = 'A'
        else:
            answer_a = example['rejected']
            answer_b = example['chosen']
            label = 'B'

        query = INSTRUCTION.format(
            question=question, 
            answer_a=answer_a, 
            answer_b=answer_b
        )

        msg = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': query}
        ]

        example['query'] = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        example['label'] = label
        
        return example

    dataset = dataset.map(process_example)
    prompts = [example['query'] for example in dataset]
    labels = [example['label'] for example in dataset]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(labels)
    results = defaultdict(dict)
    for i, output in enumerate(outputs):
        results[i]['text'] = output.outputs[0].text
        results[i]['label'] = labels[i]

    output_path = f"../pred/{args.task}"
    if args.task == 'code':
        output_path += f'-{args.subset}'
    with open(f'{output_path}.json', 'w') as f:
        print(json.dumps(results, indent=4), file=f)


if __name__ == '__main__':
    main()