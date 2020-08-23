import os
import os
import json
import pandas as pd
from functools import reduce

with open('./SETTINGS.json') as f:
    global_config = json.load(f)


def train_moiz():
        prefix_mapping = {
            'jplu/tf-xlm-roberta-large': 'roberta',
            'bert-base-multilingual-cased': 'bert',
            'xlm-mlm-100-1280': 'xlm'
        }
        for model_name in ['jplu/tf-xlm-roberta-large', 'xlm-mlm-100-1280', 'bert-base-multilingual-cased']:
            # Train the Base Model
            train_prefix = prefix_mapping[model_name]
            train_inp_dir = f'{global_config["mas_train_inp_path"]}/{train_prefix}'
            train_files= [f'{train_inp_dir}/{"train_foreign"}_p{x}.pkl' for x in list(range(0,6))] + \
                         [f'{train_inp_dir}/{"train_english"}_p{x}.pkl' for x in list(range(0,2))]
            val_file = f'{train_inp_dir}/valid_foreign_p0.pkl'
            model_resume_file = "None"
            model_save_file = f'{global_config["mas_train_out_model_path"]}/model_{train_prefix}_base.h5'
            max_epochs = 10
            patience = 3
            init_lr = 5e-6
            class_balance = 1
            class_ratio = 2
            label_smoothing = 0
            min_thresh = 0.2
            max_thresh = 0.7
            epoch_split_factor = 5

            cmd = f'python ./Code/Moiz/train.py ' \
                  f'--dev_mode={global_config["mas_train_dev_mode"]} ' \
                  f'--model_name="{model_name}" ' \
                  f'--train_files="{train_files}" ' \
                  f'--val_file="{val_file}" ' \
                  f'--model_save_file="{model_save_file}" ' \
                  f'--max_epochs={max_epochs} ' \
                  f'--patience={patience} ' \
                  f'--init_lr={init_lr} ' \
                  f'--class_balance={class_balance} ' \
                  f'--class_ratio={class_ratio} ' \
                  f'--label_smoothing={label_smoothing} ' \
                  f'--min_thresh={min_thresh} ' \
                  f'--max_thresh={max_thresh} ' \
                  f'--epoch_split_factor={epoch_split_factor} ' \
                  f'--model_resume_file="{model_resume_file}" '
            print(cmd)
            os.system(cmd)
            # Train the FOLD models
            for fold_id in range(4):
                # Train the Step #2
                train_prefix = prefix_mapping[model_name]
                train_inp_dir = f'{global_config["mas_train_inp_path"]}/{train_prefix}'
                train_parts = list(range(fold_id, fold_id + 1))
                train_files = [f'{train_inp_dir}/test_pseudo_p0.pkl'] + \
                              [f'{train_inp_dir}/{"subtitle"}_p{x}.pkl' for x in train_parts] + \
                              [f'{train_inp_dir}/{"train_foreign"}_p{x}.pkl' for x in train_parts]
                val_file = f'{train_inp_dir}/valid_foreign_p0.pkl'
                model_resume_file = f'{global_config["mas_train_out_model_path"]}/model_{train_prefix}_base.h5'
                model_save_file = f'{global_config["mas_train_out_model_path"]}/model_{train_prefix}_fold{fold_id}.h5'
                max_epochs = 5
                patience = 2
                init_lr = 5e-6
                class_balance = 1
                class_ratio = 3
                label_smoothing = 0.1
                min_thresh = 0.3
                max_thresh = 0.6
                epoch_split_factor = 2

                cmd = f'python ./Code/Moiz/train.py ' \
                      f'--dev_mode={global_config["mas_train_dev_mode"]} ' \
                      f'--model_name="{model_name}" ' \
                      f'--train_files="{train_files}" ' \
                      f'--val_file="{val_file}" ' \
                      f'--model_save_file="{model_save_file}" ' \
                      f'--max_epochs={max_epochs} ' \
                      f'--patience={patience} ' \
                      f'--init_lr={init_lr} ' \
                      f'--class_balance={class_balance} ' \
                      f'--class_ratio={class_ratio} ' \
                      f'--label_smoothing={label_smoothing} ' \
                      f'--min_thresh={min_thresh} ' \
                      f'--max_thresh={max_thresh} ' \
                      f'--epoch_split_factor={epoch_split_factor} ' \
                      f'--model_resume_file="{model_resume_file}" '

                print(cmd)
                os.system(cmd)

                # Final Step - Fine Tuning on Validation
                train_prefix = prefix_mapping[model_name]
                train_inp_dir = f'{global_config["mas_train_inp_path"]}/{train_prefix}'
                train_files = [f'{train_inp_dir}/valid_foreign_p0.pkl']
                val_file = f'{train_inp_dir}/valid_foreign_p0.pkl'
                model_resume_file = f'{global_config["mas_train_out_model_path"]}/model_{train_prefix}_fold{fold_id}.h5'
                model_save_file = f'{global_config["mas_train_out_model_path"]}/model_{train_prefix}_fold{fold_id}.h5'
                max_epochs = 1
                patience = 1
                init_lr = 5e-6
                class_balance = 0
                class_ratio = 3
                label_smoothing = 0
                min_thresh = 0.3
                max_thresh = 0.6
                epoch_split_factor = 1

                cmd = f'python ./Code/Moiz/train.py ' \
                      f'--dev_mode={global_config["mas_train_dev_mode"]} ' \
                      f'--model_name="{model_name}" ' \
                      f'--train_files="{train_files}" ' \
                      f'--val_file="{val_file}" ' \
                      f'--model_save_file="{model_save_file}" ' \
                      f'--max_epochs={max_epochs} ' \
                      f'--patience={patience} ' \
                      f'--init_lr={init_lr} ' \
                      f'--class_balance={class_balance} ' \
                      f'--class_ratio={class_ratio} ' \
                      f'--label_smoothing={label_smoothing} ' \
                      f'--min_thresh={min_thresh} ' \
                      f'--max_thresh={max_thresh} ' \
                      f'--epoch_split_factor={epoch_split_factor} ' \
                      f'--model_resume_file="{model_resume_file}" '

                print(cmd)
                os.system(cmd)

def train_igor():
	os.system('pip install torchvision > /dev/null')
	os.system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
	os.system('python pytorch-xla-env-setup.py --version 20200420 --apt-packages libomp5 libopenblas-dev ')
	os.system('pip install transformers==2.11.0 > /dev/null')
	os.system('pip install pandarallel > /dev/null')
	os.system('pip install catalyst==20.4.2 > /dev/null')
	train_list=[
            ['it','./Input/Igor/train_data_it_yandex.csv.zip','dbmdz/bert-base-italian-xxl-uncased','./Output/Models/Igor_dev/it/debug_yandex','./Input/Igor/validation.csv.zip','1'],
            ['es','./Input/Igor/train_data_es_google.csv.zip','dccuchile/bert-base-spanish-wwm-cased','./Output/Models/Igor_dev/es/debug_google','./Input/Igor/validation.csv.zip','0'],
            ['es','./Input/Igor/train_data_es_yandex.csv.zip','dccuchile/bert-base-spanish-wwm-cased','./Output/Models/Igor_dev/es/debug_yandex','./Input/Igor/validation.csv.zip','0'],
            ['fr','./Input/Igor/train_data_fr_google.csv.zip','camembert/camembert-large','./Output/Models/Igor_dev/fr/debug_google','no_val','0'],
            ['fr','./Input/Igor/train_data_fr_yandex.csv.zip','camembert/camembert-large','./Output/Models/Igor_dev/fr/debug_yandex','no_val','0'],
            ['ru','./Input/Igor/train_data_ru_google.csv.zip','DeepPavlov/rubert-base-cased-conversational','./Output/Models/Igor_dev/ru/debug_google','no_val','0'],
            ['ru','./Input/Igor/train_data_ru_yandex.csv.zip','DeepPavlov/rubert-base-cased-conversational','./Output/Models/Igor_dev/ru/debug_yandex','no_val','0'],
            ['tr','./Input/Igor/train_data_tr_google.csv.zip','dbmdz/bert-base-turkish-cased','./Output/Models/Igor_dev/tr/debug_dbmdz','./Input/Igor/validation.csv.zip','0'],
            ['tr','./Input/Igor/train_data_tr_google.csv.zip','savasy/bert-turkish-text-classification','./Output/Models/Igor_dev/tr/debug_savasy','./Input/Igor/validation.csv.zip','0']
            
	]
	for r in train_list:
		lang = r[0]
		input_file = r[1]
		backbone = r[2]
		model_file_prefix = r[3]
		val_file = r[4]
		val_tune = r[5]
		cmd = f'python ./Code/Igor/train.py --backbone="{backbone}" --model_file_prefix="{model_file_prefix}" --train_file="{input_file}" --val_file="{val_file}" --val_tune={val_tune} --os_file="./Input/Igor/633287_1126366_compressed_open-subtitles-synthesic.csv.zip" --lang="{lang}"'
		print(cmd)
		os.system(cmd)
		
def train_ujjwal():
    os.system('/bin/bash train.sh')


if __name__ == '__main__':
    train_moiz()
    os.chdir('Code/Ujjwal')
    train_ujjwal()
    os.chdir('../../')
    train_igor()
