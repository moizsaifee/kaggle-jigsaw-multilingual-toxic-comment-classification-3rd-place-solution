import os
import json
import pandas as pd
import numpy as np
import time

with open('./SETTINGS.json') as f:
    global_config = json.load(f)


def data_preproces_step1_moiz():
    """
    - This function does column rename, minor pre-procesing of the input files read from  mas_data_prep_inp_path/Raw/
    and saves in the mas_data_prep_out_path/processed/ path
    - mas_data_prep_out_path, mas_data_prep_inp_path are specified in SETTINGS.json

    Need the following files as input:
        1) Jigsaw 2018 English Train Data
        2) Jigsaw 2019 English Train Data
        3) Jigsaw 2020 Validation Data
        4) Jigsaw English -> Foreign Translated Data (Google / Yandex)
        5) Jigsaw 2020 Test Data
        6) Pseudo Labels for Test (Using Public Kernel Output, Not a Major driver of performance)
        7) Subtitles Data - pre-processed and pseudo lables created using an English trained model

    :return:
    """

    ## Train English
    print(f'{time.ctime()}: Processing Train English')
    inp_2018 = pd.read_csv(f'{global_config["mas_data_prep_inp_path"]}/Raw/jigsaw-toxic-comment-train.csv')
    inp_2018['non_toxic_label_max'] = inp_2018[
        ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].apply(lambda x: np.max(x), axis=1)
    inp_2018['toxic_float'] = np.maximum(inp_2018['toxic'], inp_2018['non_toxic_label_max'])
    inp_2018['lang'] = 'en'
    inp_2018 = inp_2018[['id', 'lang', 'comment_text', 'toxic_float']]

    inp_2019 = pd.read_csv(f'{global_config["mas_data_prep_inp_path"]}/Raw/jigsaw-unintended-bias-train.csv')
    inp_2019 = inp_2019[['id', 'comment_text', 'toxic', 'severe_toxicity',
                         'obscene', 'identity_attack', 'insult', 'threat']]
    for col in ['toxic', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']:
        inp_2019[col] = inp_2019[col].round(1)
    # If any of other label is set, count that in toxic too
    inp_2019['non_toxic_label_max'] = inp_2019[
        ['severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']].apply(lambda x: np.max(x), axis=1)
    inp_2019['toxic_float'] = np.maximum(inp_2019['toxic'], inp_2019['non_toxic_label_max'])
    inp_2019['lang'] = 'en'
    inp_2019 = inp_2019[['id', 'lang', 'comment_text', 'toxic_float']]
    train_english = pd.concat([inp_2018, inp_2019], axis=0).reset_index(drop=True)
    train_english.to_csv(f'{global_config["mas_data_prep_out_path"]}/processed/train_english.csv')

    print(f'{time.ctime()}: Processing Validation')
    ## Preprocess Valid Data
    valid = pd.read_csv(f'{global_config["mas_data_prep_inp_path"]}/Raw/validation.csv')
    valid = valid.rename(columns={'toxic': 'toxic_float'})
    valid = valid[['id', 'lang', 'comment_text', 'toxic_float']]
    valid.to_csv(f'{global_config["mas_data_prep_out_path"]}/processed/valid_foreign.csv')

    print(f'{time.ctime()}: Processing Train Foreign')
    ## Train Foreign
    train_foreign_ujjwal = pd.read_csv(f'{global_config["mas_data_prep_inp_path"]}/Processed_Ujjwal/train_foreign.csv')
    train_foreign_ujjwal['id'] = train_foreign_ujjwal['id'].astype('str')
    train_label = pd.read_csv(f'{global_config["mas_data_prep_out_path"]}/processed/train_english.csv', low_memory=False)
    train_label['id'] = train_label['id'].astype('str')
    train_foreign_ujjwal = pd.merge(train_foreign_ujjwal, train_label[['id', 'toxic_float']], on='id', how='left')
    train_foreign_ujjwal = train_foreign_ujjwal[~train_foreign_ujjwal['toxic_float'].isnull()].reset_index(drop=True)
    train_foreign_ujjwal = train_foreign_ujjwal[['id', 'lang', 'comment_text', 'toxic_float']]
    train_foreign_ujjwal.to_csv(f'{global_config["mas_data_prep_out_path"]}/processed/train_foreign.csv')

    print(f'{time.ctime()}: Processing Test Foreign')
    # Test Foreign
    test = pd.read_csv(f'{global_config["mas_data_prep_inp_path"]}/Raw/test.csv')
    test = test.rename(columns={'content': 'comment_text'})
    test = test[['id', 'lang', 'comment_text']]
    test.to_csv(f'{global_config["mas_data_prep_out_path"]}/processed/test_foreign.csv', index=False)

    print(f'{time.ctime()}: Processing Test Pseudo')
    ## Pseudo
    inp = pd.read_csv(f'{global_config["mas_data_prep_out_path"]}/processed/test_foreign.csv')
    inp_labels = pd.read_csv(f'{global_config["mas_data_prep_inp_path"]}/Raw/submission_public_lb9463.csv')
    inp = pd.merge(inp, inp_labels, on='id')
    inp['toxic_float'] = inp['toxic'].round(1)
    del inp['toxic']
    inp.to_csv(f'{global_config["mas_data_prep_out_path"]}/processed/test_pseudo.csv', index=False)

    # Subtitles Data
    print(f'{time.ctime()}: Processing Subtitles')
    inp = pd.read_csv(f'{global_config["mas_data_prep_inp_path"]}/Raw/subtitle_pseudo.csv')
    inp = inp[['id', 'lang', 'comment_text', 'toxic_float']]
    inp.to_csv(f'{global_config["mas_data_prep_out_path"]}/processed/subtitle_pseudo.csv', index=False)


def data_preproces_step2_moiz():
    """
    - This function takes as input files created in step1 and converts text into tokens and saves in appropriate model
    directory
    - Big files are chunked in files with 100K or so record so that we don't go out of memory when training

    :return:
    """
    prefix_mapping = {
        'jplu/tf-xlm-roberta-large': 'roberta',
        'bert-base-multilingual-cased': 'bert',
        'xlm-mlm-100-1280': 'xlm'
    }
    in_file_type = 'train'
    long_comment_action = 'ignore'

    for model_name in ['bert-base-multilingual-cased', 'jplu/tf-xlm-roberta-large', 'xlm-mlm-100-1280']:
        model_prefix = prefix_mapping[model_name]
        data_tuples = [
            ('train_foreign.csv', 'train_foreign', 1, 6),
            ('subtitle_pseudo.csv', 'subtitle', 1, 4),
            ('train_english.csv', 'train_english', 1, 2),
            ('test_pseudo.csv', 'test_pseudo', 0, 0),
            ('valid_foreign.csv', 'valid_foreign', 0, 0)
        ]
        for data_tuple in data_tuples:
            in_file, out_file, should_chunk, max_chunk = data_tuple
            print(f'{time.ctime()} Processing {in_file}')
            cmd = f'python ./Code/Moiz/data_prep.py ' \
                  f'--in_file="{global_config["mas_data_prep_out_path"]}/processed/{in_file}" ' \
                  f'--in_file_type="{in_file_type}" ' \
                  f'--model_name="{model_name}" ' \
                  f'--should_chunk={should_chunk} ' \
                  f'--max_chunk={max_chunk} ' \
                  f'--long_comment_action="{long_comment_action}" ' \
                  f'--out_dir="{global_config["mas_data_prep_out_path"]}/{model_prefix}/" ' \
                  f'--out_file="{out_file}" '
            print(cmd)
            os.system(cmd)


if __name__ == '__main__':
    data_preproces_step1_moiz()
    data_preproces_step2_moiz()