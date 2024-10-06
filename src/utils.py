import torch
from transformers import AutoTokenizer


DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information."
DEFAULT_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"


def set_seed(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_chat_template(
    example, 
    tokenizer,
    construct_msg=True,
    system_prompt=DEFAULT_SYSTEM_PROMPT, 
    inst_key="instruction", 
    input_key=None,
    output_key="response",
    result_key="text"
):
    if construct_msg:
        if (not input_key) or example[input_key] == "":
            user_input = f"{example[inst_key]}"
        else:
            user_input = f"{example[inst_key]}\n{example[input_key]}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": example[output_key]}
        ]
    else:
        messages = [{"role": "system", "content": system_prompt}] + example[result_key]
    
    example[result_key] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example


def format_chosen_rejected(
    dataset, 
    tokenizer: AutoTokenizer, 
    inst_key: str, 
    input_key: str = None, 
    construct_msg: bool = True
):
    """
    args:
        inst_key - key for input prompt
        input_key - key for user input
        construct_msg - set to true if chosen/rejected is already in the format that chat template can be applied
    additional stuff:
        output_key - key for model response
        result_key - key for storing the processed string
    """
    for key in ["chosen", "rejected"]:
        fn_kwargs = {
            "tokenizer": tokenizer, 
            "inst_key": inst_key, 
            "input_key": input_key,
            "output_key": key,
            "result_key": key,
            "construct_msg": construct_msg
        }
        dataset = dataset.map(apply_chat_template, fn_kwargs=fn_kwargs)
    
    return dataset


def default_func(dataset):
    return dataset # do nothing


def preprocess_shp(dataset, tokenizer: AutoTokenizer, score_ratio: float):
    def create_chosen_rejected(example):
        if example['score_A'] > example['score_B']:
            example['chosen'] = example['human_ref_A']
            example['rejected'] = example['human_ref_B']
        else:
            example['chosen'] = example['human_ref_B']
            example['rejected'] = example['human_ref_A']
        
        return example
    
    # follow the training tips provided by the author
    dataset = dataset.filter(lambda example: example['score_ratio'] >= score_ratio)
    dataset = dataset.map(create_chosen_rejected)
    dataset = format_chosen_rejected(dataset, tokenizer, inst_key="history")

    return dataset


def preprocess_metamath(dataset, tokenizer: AutoTokenizer):
    dataset = format_chosen_rejected(dataset, tokenizer, inst_key="prompt")

    return dataset


def preprocess_distilabel_math(dataset, tokenizer: AutoTokenizer):
    def create_chosen_reject(example):
        example['chosen'] = example['chosen_response']
        example['rejected'] = example['rejected_response']

        return example

    dataset = dataset.map(create_chosen_reject)
    dataset = format_chosen_rejected(dataset, tokenizer, inst_key="instruction")

    return dataset


def preprocess_alpaca(dataset, tokenizer: AutoTokenizer):
    def create_chosen_reject(example):
        if example['preference'] == 1:
            example['chosen'] = example['output_1']
            example['rejected'] = example['output_2']
        else:
            example['chosen'] = example['output_2']
            example['rejected'] = example['output_1']
        return example
    
    def filter_func(example):
        ban_list = ['code', 'function', 'programming', 'python', 'cpp', 'js', 'java', 'rust', 'c++', 'script', 'algorithm', 'data structure']
        for ban_word in ban_list:
            if ban_word in example['instruction']:
                return False
        return True

    dataset = dataset.filter(filter_func) # filter coding data
    dataset = dataset.map(create_chosen_reject)
    dataset = format_chosen_rejected(dataset, tokenizer, "instruction", "input")

    return dataset


def preprocess_pku_alignment(dataset, tokenizer: AutoTokenizer):
    def create_chosen_reject(example):
        if example['safer_response_id'] == 0:
            example['chosen'] = example['response_0']
            example['rejected'] = example['response_1']
        else:
            example['chosen'] = example['response_1']
            example['rejected'] = example['response_0']
        return example
    
    dataset = dataset.filter(lambda example: example['is_response_0_safe'] != example['is_response_1_safe'])
    dataset = dataset.map(create_chosen_reject)
    dataset = format_chosen_rejected(dataset, tokenizer, "prompt")

    return dataset


def preprocess_summarization(dataset, tokenizer: AutoTokenizer):
    def process(example):
        example['chosen'] = tokenizer.apply_chat_template(example['chosen'], tokenize=False)
        example['rejected'] = tokenizer.apply_chat_template(example['rejected'], tokenize=False)
        return example
    dataset = dataset.map(process)
    return dataset


def preprocess_ultrafeedback(dataset, tokenizer: AutoTokenizer):
    dataset = format_chosen_rejected(dataset, tokenizer, inst_key="prompt", construct_msg=False)
    return dataset


def preprocess_ultrafeedback_alpaca(dataset, tokenizer: AutoTokenizer):
    prompt_template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    )
    prompt_template = ''.join(prompt_template)
    def format_prompt(example):
        assert len(example['chosen']) == len(example['rejected']) and len(example['chosen']) == 2
        chosen_text = prompt_template.format(
            instruction=example['chosen'][0]['content'], 
            response=example['chosen'][1]['content']
        )
        rejected_text = prompt_template.format(
            instruction=example['rejected'][0]['content'], 
            response=example['rejected'][1]['content']
        )

        example['chosen'] = chosen_text
        example['rejected'] = rejected_text
        return example

    dataset = dataset.map(format_prompt)
    return dataset


def preprocess_hh(dataset, tokenizer: AutoTokenizer):
    # We modified the code from: https://huggingface.co/datasets/trl-internal-testing/hh-rlhf-trl-style/blob/0.1.0/anthropic_hh.py
    # GPT-4 generated 😄 Define a function to process the input and extract the dialogue into structured format
    def extract_dialogue(input_text):
        # Split the input by lines and initialize variables
        lines = input_text.strip().split('\n\n')
        dialogue_list = []
        
        # Iterate through each line and extract the dialogue
        for line in lines:
            # Check if the line starts with "Human" or "Assistant" and split accordingly
            if line.startswith("Human:"):
                role = "user"
                content = line.replace("Human: ", "").strip()
            elif line.startswith("Assistant:"):
                role = "assistant"
                content = line.replace("Assistant: ", "").strip()
            else:
                # If the line doesn't start with "Human" or "Assistant", it's part of the previous message's content
                # Append it to the last message's content
                dialogue_list[-1]['content'] += "\n\n" + line.strip()
                continue

            # Append the extracted dialogue piece to the list
            dialogue_list.append({"role": role, "content": content})
        
        return dialogue_list

    def parse_conv(example):
        for key in ['chosen', 'rejected']:
            example[key] = extract_dialogue(example[key])
            example[key] = tokenizer.apply_chat_template(example[key], tokenize=False)

        return example
    
    dataset = dataset.map(parse_conv, load_from_cache_file=False)

    return dataset


def preprocess_autoj(dataset, tokenizer: AutoTokenizer):
    dataset = dataset.filter(lambda example: example['label'] != 2) # filter out the tie cases

    def categorize_scenario(example):
        if example['scenario'] in ['math_reasoning', 'solving_exam_question_with_math']:
            example['category'] = 'Math'
        elif example['scenario'] in ['code_simplification', 'code_generation', 'explaining_code', 'code_to_code_translation', 'code_correction_rewriting']:
            example['category'] = 'Code'
        else:
            example['category'] = 'Other'

        return example

    dataset = dataset.map(categorize_scenario)
    dataset = dataset.filter(lambda example: example['category'] in ['Math']) # we only use math & code subset for fine-tuning

    def create_chosen_reject(example):
        if example['label'] == 0:
            example['chosen'] = example['response 1']
            example['rejected'] = example['response 2']
        else:
            example['chosen'] = example['response 2']
            example['rejected'] = example['response 1']

        return example
    
    dataset = dataset.map(create_chosen_reject)
    dataset = format_chosen_rejected(dataset, tokenizer, "prompt")

    return dataset
