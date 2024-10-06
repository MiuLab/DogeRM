import json
import argparse

from bon_utils import find_number, maybe_remove_comma, most_frequent_element


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../bon_output/gsm8k-llama2-chat.json")
    parser.add_argument('--best_of', type=int, default=16)

    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        data = json.load(f)
    data = list(data.values())

    # parse answer in each response
    model_response = [
        [maybe_remove_comma(find_number(ans)) for ans in output['answer']]
          for output in data
    ]
    ground_truth = [output['ground_truth'][0] for output in data]
    curr = 1
    while curr <= args.best_of:
        # voting
        consistent_response = [most_frequent_element(output[:curr]) for output in model_response]
        # calculate acc.
        correct_cnt = 0
        for pred, gd in zip(consistent_response, ground_truth):
            if pred != '' and float(pred) == float(gd):
                correct_cnt += 1
        acc = correct_cnt / len(consistent_response)

        print(f'Self-consistency@{curr}: {acc}')

        curr = curr << 1


if __name__ == '__main__':
    main()