# https://www.kaggle.com/moizsaifee/jigsaw-train-v10-step2-mcp-submission
import sys
sys.path.append("./Code/Moiz/")

from model_utils import *
import argparse

logger = init_logger()
set_seed()

"""
    This script takes the following thing as input:
        1) An Input file to be scored
        2) An Input model used for scoring
    and produces:
        1) Scored input file
        
    Parameters:
        - dev_mode: for debugging - for debugging purpose, if switched on, then only some specific IDs are scored 
        - model_name: one of 'jplu/tf-xlm-roberta-large' / 'xlm-mlm-100-1280' / 'bert-base-multilingual-cased'
        - model_file: The name of saved model bin file, artificat of the training process
        - in_file: The name of (pre-processed) input file to be scored, need to run data prep to generate thi from raw
        - in_file_type: one of "extra" or "foreign" / "english" - slightly different ways in which prob across multiple obs 
            of the same id are handled - Max vs avg - that's the only difference
        - out_file: The file (full path) in which the results would be written to
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_mode', type=int, default=0)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_file', type=str)

    parser.add_argument('--in_file', type=str)
    parser.add_argument('--in_file_type', type=str)
    parser.add_argument('--out_file', type=str)

    parsed_args = parser.parse_args()

    config = {
        'max_len': 192,
        'batch_size': 16,
        'model_name': parsed_args.model_name,
        'dev_mode': parsed_args.dev_mode,
    }

    solver = Solver(logger, config)
    if parsed_args.in_file_type == 'extra':
        solver.score_test_extra(
            parsed_args.in_file,
            parsed_args.model_file,
            parsed_args.out_file
        )
    else:
        solver.score_test_split(
            parsed_args.in_file,
            parsed_args.model_file,
            parsed_args.out_file
        )



