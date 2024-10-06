import argparse
import torch

from transformers import (
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM, 
    AutoModel, 
    AutoTokenizer,
)


def merge_embed(args, model_seq, model_lm, tokenizer_seq, tokenizer_lm):
    vocab_seq = tokenizer_seq.get_vocab()

    # if the language model did not resize the embedding, we manually resize the embedding layer
    if len(tokenizer_lm) != model_lm.get_input_embeddings().num_embeddings:
        model_lm.resize_token_embeddings(len(tokenizer_lm))
    
    vocab_lm = dict(sorted(tokenizer_lm.get_vocab().items(), key=lambda item: item[1]))
    embeddings_seq = model_seq.get_input_embeddings().weight.detach().clone()
    embeddings_lm = model_lm.get_input_embeddings().weight.detach().clone()

    new_vocab = vocab_seq.copy()
    new_embeddings = embeddings_lm.clone()
    for token, index_lm in vocab_lm.items():
        if token in new_vocab:
            # Calculate weighted average for common tokens
            index_seq = new_vocab[token]
            assert index_lm == index_seq
            new_embeddings[index_lm] = (args.seq_weight * embeddings_seq[index_seq] + args.lm_weight * embeddings_lm[index_lm])
        else:
            # Add unique tokens from model_lm
            print(f'add {token} : {index_lm} from embedding_lm')
            new_index = len(new_vocab)
            new_vocab[token] = new_index
            assert torch.equal(new_embeddings[index_lm], embeddings_lm[index_lm])
            # new_embeddings = torch.cat([new_embeddings, embeddings_lm[index_lm].unsqueeze(0)], dim=0)
            # Update tokenizer_seq
            tokenizer_seq.add_tokens(token, special_tokens=(token in tokenizer_lm.special_tokens_map.values() or token in tokenizer_lm.additional_special_tokens))

    model_seq.get_input_embeddings().weight = torch.nn.Parameter(new_embeddings)
    model_seq.resize_token_embeddings(len(tokenizer_seq))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_model', type=str, required=True)
    parser.add_argument('--seq_weight', type=float, default=0.5)
    parser.add_argument('--lm_model', type=str, required=True)
    parser.add_argument('--lm_weight', type=float, default=0.5)
    parser.add_argument('--use_lm_embed', action="store_true")
    parser.add_argument('--output_path', type=str, default="models/llama2")
    args = parser.parse_args()

    # assert args.seq_weight + args.lm_weight == 1.0, "weights should sum up to 1."

    # Load your pretrained models
    seq_builder = AutoModelForSequenceClassification.from_pretrained if "Eurus" not in args.seq_model else AutoModel.from_pretrained
    model_seq = AutoModelForSequenceClassification.from_pretrained(args.seq_model, torch_dtype=torch.float32, trust_remote_code=True)
    model_lm = AutoModelForCausalLM.from_pretrained(args.lm_model, torch_dtype=torch.float32)
    # print(model_lm)

    tokenizer_seq = AutoTokenizer.from_pretrained(args.seq_model)
    tokenizer_lm = AutoTokenizer.from_pretrained(args.lm_model)

    # Define the weights for weighted sum
    alpha = args.seq_weight  # weight for the sequence classification model
    beta = args.lm_weight   # weight for the causal language model

    # Perform the weighted sum of parameters
    for (name_seq, param_seq), (name_lm, param_lm) in zip(model_seq.named_parameters(), model_lm.named_parameters()):
        if 'embed' in name_seq or 'embed' in name_lm:
            merge_embed(args, model_seq, model_lm, tokenizer_seq, tokenizer_lm)
            continue
        if 'score' in name_seq or 'lm_head' in name_lm:
            # Skip the linear head parts
            continue
        # Check if both names are identical to ensure corresponding parameters
        if name_seq == name_lm:
            # Perform the weighted sum
            weighted_sum_param = alpha * param_seq.data + beta * param_lm.data
            # Assign the new parameters back to the sequence classification model
            param_seq.data.copy_(weighted_sum_param)

    del model_lm
    # Save the updated sequence classification model
    model_seq.save_pretrained(args.output_path)
    tokenizer_seq.save_pretrained(args.output_path)

if __name__ == '__main__':
    main()
