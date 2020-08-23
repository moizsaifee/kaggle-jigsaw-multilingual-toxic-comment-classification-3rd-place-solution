import sys
sys.path.append("./Code/Moiz/")

import argparse
from data_utils import *

def data_prep(in_file, file_type, model_name, should_chunk, max_chunk,
              long_comment_action, out_dir, out_file_prefix):
    """
        Takes an input file, model type as input and produces the
        the output file with tokenized text

    :return:
    """
    inp = pd.read_csv(in_file)
    print(f'Preparing Input: {in_file}')

    if file_type == 'test':
        inp = inp[['id', 'lang', 'comment_text']]
    elif file_type == 'train':
        inp = inp[['id', 'lang', 'comment_text', 'toxic_float']]
    else:
        raise NotImplementedError

    process_file(
        192, model_name,
        inp, file_type, out_dir, out_file_prefix,
        should_chunk = should_chunk, chunk_size=100000,
        long_comment_action=long_comment_action, max_chunk=max_chunk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str)
    parser.add_argument('--in_file_type', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--should_chunk', type=int)
    parser.add_argument('--max_chunk', type=int, default=0)
    parser.add_argument('--long_comment_action', type=str)

    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--out_file_prefix', type=str)

    parsed_args = parser.parse_args()

    should_chunk = parsed_args.should_chunk != 0

    if parsed_args.max_chunk == 0:
        max_chunk = None
    else:
        max_chunk = parsed_args.max_chunk

    data_prep(
        parsed_args.in_file, parsed_args.in_file_type, parsed_args.model_name,
        should_chunk, max_chunk,
        parsed_args.long_comment_action, parsed_args.out_dir, parsed_args.out_file_prefix
    )