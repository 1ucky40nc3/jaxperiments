from typing import List

import argparse

import pandas as pd


def load_names(path: str = 'names.txt') -> List[str]:
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [line.strip() for line in lines]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, default='names.txt')
    args = parser.parse_args()

    names = load_names(args.dataset_file)
    print(f'Loaded names from the given file. Here is a subset of the data: \n{pd.DataFrame(names)}')

    tokens = set([token for name in names for token in name])
    tokens = sorted(list(tokens))
    print(f'The following tokens will be used: {tokens}')
    token_to_id = {i: t for i, t in enumerate(tokens)}
    id_to_token = {t: i for i, t in token_to_id.items()}

    


if __name__ == '__main__':
    main()