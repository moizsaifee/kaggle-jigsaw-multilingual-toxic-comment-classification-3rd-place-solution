# https://www.kaggle.com/moizsaifee/jigsaw-train-v10-step2-mcp
# https://www.kaggle.com/moizsaifee/jigsaw-train-v10-step2-mcp-variant1
# https://www.kaggle.com/moizsaifee/jigsaw-train-v10-step2-mcp-variant2
# https://www.kaggle.com/moizsaifee/jigsaw-train-v10-step2-mcp-variant3

import sys
sys.path.append("./Code/Moiz/")
from model_utils import *
import ast

import argparse
logger = init_logger()
set_seed()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_mode', type=int, default=0)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--train_files', type=str)
    parser.add_argument('--val_file', type=str)

    parser.add_argument('--model_save_file', type=str)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--init_lr', type=float)
    parser.add_argument('--class_balance', type=int)
    parser.add_argument('--class_ratio', type=int)
    parser.add_argument('--label_smoothing', type=float)
    parser.add_argument('--min_thresh', type=float)
    parser.add_argument('--max_thresh', type=float)
    parser.add_argument('--epoch_split_factor', type=int)
    parser.add_argument('--model_resume_file', type=str)

    parsed_args = parser.parse_args()
    print(parsed_args)
    config = {
        'max_len': 192,
        'batch_size': 16,
        'dev_mode': parsed_args.dev_mode,
        'model_name': parsed_args.model_name,
    }
    if parsed_args.model_resume_file == "None":
        model_resume_file = None
    else:
        model_resume_file = parsed_args.model_resume_file

    solver = Solver(logger, config)
    solver.train(
        train_files = ast.literal_eval(parsed_args.train_files),
        val_file = parsed_args.val_file,
        model_save_file = parsed_args.model_save_file,
        max_epochs = parsed_args.max_epochs,
        patience=parsed_args.patience,
        init_lr=parsed_args.init_lr,
        class_balance=parsed_args.class_balance,
        class_ratio=parsed_args.class_ratio,
        label_smoothing=parsed_args.label_smoothing,
        min_thresh=parsed_args.min_thresh,
        max_thresh=parsed_args.max_thresh,
        epoch_split_factor=parsed_args.epoch_split_factor,
        model_resume_file=model_resume_file
    )
