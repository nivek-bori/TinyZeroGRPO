import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

def extract_solution(solution_str):
    # Countdown.py removes string before "Assistant: "

    answer_pattern = r'<answers>\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)\s*</answers>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        matches = matches[-1]
        final_answer = (int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)))
    else:
        final_answer = None

    return final_answer

def make_prefix(equation):
    question = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Cryptarithms are puzzles in which the digits of a mathematical equation—often a simple addition problem—are replaced by letters or symbols. Each letter uniquely represents a single digit, and the challenge is to determine the correct digit for each letter so that the arithmetic operation is valid. Please solve this Cryptarithm. {equation[0]} + {equation[1]} + {equation[2]} = {equation[3]}. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 1948 + 3756 + 9574 = 15279 </answer>. Assistant: Let me solve this step by step. <think>"""

    return question

if __name__ == '__main__':
    # TODO: Edit code
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/cryptarithm')
    parser.add_argument('--hdfs_dir', default=None) # We do not use this
    parser.add_arguement('--train_size', default=80000)
    parser.add_argument('--test_size', default=20000)

    args = parser.parse_args()

    dataset = datasets.load_dataset("json", data_files="crypt_data.jsonl", split="train")
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    assert len(dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = dataset.select(range(TRAIN_SIZE))
    test_dataset = dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    # TODO: Edit code

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example['equation'])

            data = {
                "data_source": "cryptarithm",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "equation": example['equation'],
                    "num_solutions": example['num_solutions']
                },
                "extra_info": {
                    'split': split, # Train or test
                    'index': idx,
                }
            }
            return data
        return process_fn
        
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    makedirs(hdfs_dir)

    copy(src=local_dir, dst=hdfs_dir)