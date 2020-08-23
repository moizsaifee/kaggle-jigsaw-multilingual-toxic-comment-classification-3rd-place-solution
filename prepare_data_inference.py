import os
import json


with open('./SETTINGS.json') as f:
    global_config = json.load(f)


def data_prep_moiz():
    data_typles = [
        ('test_foreign.csv', 'jplu/tf-xlm-roberta-large', 'split', 'test_roberta_foreign_split'),
        ('test_english.csv', 'jplu/tf-xlm-roberta-large', 'ignore', 'test_roberta_english'),
        ('test_extra.csv', 'jplu/tf-xlm-roberta-large', 'ignore', 'test_roberta_extra'),

        ('test_foreign.csv', 'bert-base-multilingual-cased', 'ignore', 'test_bert_foreign_split'),
        ('test_english.csv', 'bert-base-multilingual-cased', 'ignore', 'test_bert_english'),
        ('test_extra.csv', 'bert-base-multilingual-cased', 'ignore', 'test_bert_extra'),

        ('test_foreign.csv', 'xlm-mlm-100-1280', 'ignore', 'test_xlm_foreign_split'),
        ('test_english.csv', 'xlm-mlm-100-1280', 'ignore', 'test_xlm_english'),
        ('test_extra.csv', 'xlm-mlm-100-1280', 'ignore', 'test_xlm_extra'),

    ]
    in_file_type= 'test'
    should_chunk = 0
    max_chunk = 0

    for data_tuple in data_typles:
        in_file, model_name, long_comment_action, out_file = data_tuple
        cmd = f'python ./Code/Moiz/data_prep.py ' \
              f'--in_file="{global_config["mas_data_prep_inp_path"]}/Processed_Ujjwal/{in_file}" ' \
              f'--in_file_type="{in_file_type}" ' \
              f'--model_name="{model_name}" ' \
              f'--should_chunk={should_chunk} ' \
              f'--max_chunk={max_chunk} ' \
              f'--long_comment_action="{long_comment_action}" ' \
              f'--out_dir="{global_config["mas_predict_inp_path"]}" ' \
              f'--out_file="{out_file}" '
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    data_prep_moiz()
