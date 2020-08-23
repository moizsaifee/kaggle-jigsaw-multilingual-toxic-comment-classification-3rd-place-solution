import os
from glob import glob
import json
import pandas as pd
from functools import reduce

with open('./SETTINGS.json') as f:
    global_config = json.load(f)


def predict_moiz():
    """
    Set up the tuples of input arguments to be able to score all the desired folds of the model. This function
    expects the items listed under input to be present in the indicated directories

    Note: This function executes the python scripts through system call, so if any input / output / model is missing
    the code will not halt and continue to run - error messages would be displaced on screen though.

        - Input:
            Models: (location - at <mas_predict_model_path> SETTINGS.json)
                Roberta - model_roberta_base.h5, model_roberta_fold0-3
                Bert - model_bert_base.h5, model_bert_fold0-3
                XLM - model_xlm_base.h5, model_xlm_fold0-2
            Scoring Input: (location - at <mas_predict_in_path> SETTINGS.json)
                Preprocessed {Foreign, English, Foreign -> Foreign} x {Roberta, Bert, XLM} tokenized inputs
        - Output: (location - at <mas_predict_out_path> SETTINGS.json)
            - pred_roberta_*
            - pred_bert_*
            - pred_xlm_*
    :return: nothing, everything written to file
    """

    score_tuples = [
        
        # Roberta - English model, just 1 fold
        ("test_roberta_english_p0.pkl", "english", "jplu/tf-xlm-roberta-large", "model_roberta_base.h5", "pred_roberta_english.csv"),
        
        # Roberta - Foreign, 4 fold
        ("test_roberta_foreign_split_p0.pkl", "foreign", "jplu/tf-xlm-roberta-large", "model_roberta_fold0.h5", "pred_roberta_foreign_fold0.csv"),
        ("test_roberta_foreign_split_p0.pkl", "foreign", "jplu/tf-xlm-roberta-large", "model_roberta_fold1.h5", "pred_roberta_foreign_fold1.csv"),
        ("test_roberta_foreign_split_p0.pkl", "foreign", "jplu/tf-xlm-roberta-large", "model_roberta_fold2.h5", "pred_roberta_foreign_fold2.csv"),
        ("test_roberta_foreign_split_p0.pkl", "foreign", "jplu/tf-xlm-roberta-large", "model_roberta_fold3.h5", "pred_roberta_foreign_fold3.csv"),
        
        # Roberta extra (translated foreign) - 4 fold
        ("test_roberta_extra_p0.pkl", "extra", "jplu/tf-xlm-roberta-large", "model_roberta_fold0.h5", "pred_roberta_extra_fold0.csv"),
        ("test_roberta_extra_p0.pkl", "extra", "jplu/tf-xlm-roberta-large", "model_roberta_fold1.h5", "pred_roberta_extra_fold1.csv"),
        ("test_roberta_extra_p0.pkl", "extra", "jplu/tf-xlm-roberta-large", "model_roberta_fold2.h5", "pred_roberta_extra_fold2.csv"),
        ("test_roberta_extra_p0.pkl", "extra", "jplu/tf-xlm-roberta-large", "model_roberta_fold3.h5", "pred_roberta_extra_fold3.csv"),
        
        # Bert - English model, just 1 fold
        ("test_bert_english_p0.pkl", "english", "bert-base-multilingual-cased", "model_bert_base.h5", "pred_bert_english.csv"),
        
        # Bert - Foreign, 4 fold
        ("test_bert_foreign_split_p0.pkl", "foreign", "bert-base-multilingual-cased", "model_bert_fold0.h5", "pred_bert_foreign_fold0.csv"),
        ("test_bert_foreign_split_p0.pkl", "foreign", "bert-base-multilingual-cased", "model_bert_fold1.h5", "pred_bert_foreign_fold1.csv"),
        ("test_bert_foreign_split_p0.pkl", "foreign", "bert-base-multilingual-cased", "model_bert_fold2.h5", "pred_bert_foreign_fold2.csv"),
        ("test_bert_foreign_split_p0.pkl", "foreign", "bert-base-multilingual-cased", "model_bert_fold3.h5", "pred_bert_foreign_fold3.csv"),
        
        # Bert extra (translated foreign) - 4 fold
        ("test_bert_extra_p0.pkl", "extra", "bert-base-multilingual-cased", "model_bert_fold0.h5", "pred_bert_extra_fold0.csv"),
        ("test_bert_extra_p0.pkl", "extra", "bert-base-multilingual-cased", "model_bert_fold1.h5", "pred_bert_extra_fold1.csv"),
        ("test_bert_extra_p0.pkl", "extra", "bert-base-multilingual-cased", "model_bert_fold2.h5", "pred_bert_extra_fold2.csv"),
        ("test_bert_extra_p0.pkl", "extra", "bert-base-multilingual-cased", "model_bert_fold3.h5", "pred_bert_extra_fold3.csv"),

        # XLM - English model, just 1 fold
        ("test_xlm_english_p0.pkl", "english", "xlm-mlm-100-1280", "model_xlm_base.h5", "pred_xlm_english.csv"),

        # XLM - Foreign,  ~/.tr 3 fold
        ("test_xlm_foreign_split_p0.pkl", "foreign", "xlm-mlm-100-1280", "model_xlm_fold0.h5", "pred_xlm_foreign_fold0.csv"),
        ("test_xlm_foreign_split_p0.pkl", "foreign", "xlm-mlm-100-1280", "model_xlm_fold1.h5", "pred_xlm_foreign_fold1.csv"),
        ("test_xlm_foreign_split_p0.pkl", "foreign", "xlm-mlm-100-1280", "model_xlm_fold2.h5", "pred_xlm_foreign_fold2.csv"),

    ]
    for score_tuple in score_tuples:
        in_file, in_file_type, model_name, model_file, out_file = score_tuple
        cmd = f'python ./Code/Moiz/predict.py ' \
              f'--dev_mode={global_config["mas_predict_dev_mode"]} ' \
              f'--in_file="{global_config["mas_predict_inp_path"]}/{in_file}" ' \
              f'--in_file_type="{in_file_type}" ' \
              f'--out_file="{global_config["mas_predict_out_path"]}/{out_file}" ' \
              f'--model_name="{model_name}" ' \
              f'--model_file={global_config["mas_predict_model_path"]}/{model_file} '
        print(cmd)
        os.system(cmd)


def belnd_roberta_moiz():
    print('Blending Roberta')
    # https://www.kaggle.com/moizsaifee/jigsaw-train-v10-step2-mcp-submission
    wd = os.getcwd()
    os.chdir("./Output/Predictions/Moiz/")
    res1 = pd.read_csv('./pred_roberta_foreign_fold0.csv').rename(
        columns={'toxic': 'toxic_f1'})
    res2 = pd.read_csv('./pred_roberta_english.csv').rename(
        columns={'toxic': 'toxic_e1'})
    res3 = pd.read_csv('./pred_roberta_extra_fold0.csv').rename(
        columns={'toxic': 'toxic_ex1'})

    res4 = pd.read_csv('./pred_roberta_foreign_fold1.csv').rename(
        columns={'toxic': 'toxic_f2'})
    res5 = pd.read_csv('./pred_roberta_english.csv').rename(
        columns={'toxic': 'toxic_e2'})
    res6 = pd.read_csv('./pred_roberta_extra_fold1.csv').rename(
        columns={'toxic': 'toxic_ex2'})

    res7 = pd.read_csv('./pred_roberta_foreign_fold2.csv').rename(
        columns={'toxic': 'toxic_f3'})
    res8 = pd.read_csv('./pred_roberta_english.csv').rename(
        columns={'toxic': 'toxic_e3'})
    res9 = pd.read_csv('./pred_roberta_extra_fold2.csv').rename(
        columns={'toxic': 'toxic_ex3'})

    res10 = pd.read_csv('./pred_roberta_foreign_fold3.csv').rename(
        columns={'toxic': 'toxic_f4'})
    res11 = pd.read_csv('./pred_roberta_english.csv').rename(
        columns={'toxic': 'toxic_e4'})
    res12 = pd.read_csv('./pred_roberta_extra_fold3.csv').rename(
        columns={'toxic': 'toxic_ex4'})

    res = reduce(lambda x, y: pd.merge(x, y, on='id'),
                 [res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12])

    res.head()

    K = 1. / 2
    res['toxic'] = ((
        (0.8 * res['toxic_f1'] + 0.2 * (0.5 * res['toxic_e1'] + 0.5 * res['toxic_ex1'])) ** K +
        (0.8 * res['toxic_f2'] + 0.2 * (0.5 * res['toxic_e2'] + 0.5 * res['toxic_ex2'])) ** K +
        (0.8 * res['toxic_f3'] + 0.2 * (0.5 * res['toxic_e3'] + 0.5 * res['toxic_ex3'])) ** K +
        (0.8 * res['toxic_f4'] + 0.2 * (0.5 * res['toxic_e4'] + 0.5 * res['toxic_ex4'])) ** K
    ) / 4) ** (1 / K)
    os.chdir(wd)
    res.to_csv('./Output/Predictions/Moiz/submission_Roberta.csv')
    return res[['id', 'toxic']]


def blend_bert_moiz():
    print('Blending Bert')

    wd = os.getcwd()
    os.chdir("./Output/Predictions/Moiz/")
    # https://www.kaggle.com/moizsaifee/bert-jigsaw-train-v10-step2-mcp-submission
    res1 = pd.read_csv('./pred_bert_foreign_fold0.csv').rename(
        columns={'toxic': 'toxic_f1'})
    res2 = pd.read_csv('./pred_bert_english.csv').rename(
        columns={'toxic': 'toxic_e1'})
    res3 = pd.read_csv('./pred_bert_extra_fold0.csv').rename(
        columns={'toxic': 'toxic_ex1'})

    res4 = pd.read_csv('./pred_bert_foreign_fold1.csv').rename(
        columns={'toxic': 'toxic_f2'})
    res5 = pd.read_csv('./pred_bert_english.csv').rename(
        columns={'toxic': 'toxic_e2'})
    res6 = pd.read_csv('./pred_bert_extra_fold0.csv').rename(
        columns={'toxic': 'toxic_ex2'})

    res7 = pd.read_csv('./pred_bert_foreign_fold2.csv').rename(
        columns={'toxic': 'toxic_f3'})
    res8 = pd.read_csv('./pred_bert_english.csv').rename(
        columns={'toxic': 'toxic_e3'})
    res9 = pd.read_csv('./pred_bert_extra_fold0.csv').rename(
        columns={'toxic': 'toxic_ex3'})

    res10 = pd.read_csv('./pred_bert_foreign_fold3.csv').rename(
        columns={'toxic': 'toxic_f4'})
    res11 = pd.read_csv('./pred_bert_english.csv').rename(
        columns={'toxic': 'toxic_e4'})
    res12 = pd.read_csv('./pred_bert_extra_fold0.csv').rename(
        columns={'toxic': 'toxic_ex4'})

    res = reduce(lambda x, y: pd.merge(x, y, on='id'),
                 [res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12])  # ]

    K = 1. / 2
    res['toxic'] = ((
        (0.8 * res['toxic_f1'] + 0.2 * (0.5 * res['toxic_e1'] + 0.5 * res['toxic_ex1'])) ** K +
        (0.8 * res['toxic_f2'] + 0.2 * (0.5 * res['toxic_e2'] + 0.5 * res['toxic_ex2'])) ** K +
        (0.8 * res['toxic_f3'] + 0.2 * (0.5 * res['toxic_e3'] + 0.5 * res['toxic_ex3'])) ** K +
        (0.8 * res['toxic_f4'] + 0.2 * (0.5 * res['toxic_e4'] + 0.5 * res['toxic_ex4'])) ** K
                    ) / 4) ** (1 / K)
    os.chdir(wd)
    return res[['id', 'toxic']]


def blend_xlm_moiz():
    print('Blending XLM')
    wd = os.getcwd()
    os.chdir("./Output/Predictions/Moiz/")
    # https://www.kaggle.com/moizsaifee/xlm-jigsaw-train-v10-step2-mcp-submission
    res1 = pd.read_csv('./pred_xlm_foreign_fold0.csv').rename(
        columns={'toxic': 'toxic_f1'})
    res2 = pd.read_csv('./pred_xlm_english.csv').rename(
        columns={'toxic': 'toxic_e1'})

    res4 = pd.read_csv('./pred_xlm_foreign_fold1.csv').rename(
        columns={'toxic': 'toxic_f2'})
    res5 = pd.read_csv('./pred_xlm_english.csv').rename(
        columns={'toxic': 'toxic_e2'})

    res7 = pd.read_csv('./pred_xlm_foreign_fold2.csv').rename(
        columns={'toxic': 'toxic_f3'})
    res8 = pd.read_csv('./pred_xlm_english.csv').rename(
        columns={'toxic': 'toxic_e3'})
    res = reduce(lambda x, y: pd.merge(x, y, on='id'), [res1, res2, res4, res5, res7, res8])

    K=1/2
    res['toxic'] = ((
        (0.8*res['toxic_f1'] + 0.2*(res['toxic_e1']))**K +
        (0.8*res['toxic_f2'] + 0.2*(res['toxic_e2']))**K +
        (0.8*res['toxic_f3'] + 0.2*(res['toxic_e3']))**K
    )/3)**(1/K)
    # res['toxic'] = res['toxic_x']
    os.chdir(wd)
    return res[['id', 'toxic']]


def blend_moiz():
    # https://www.kaggle.com/moizsaifee/moiz-submissions-final-blender?scriptVersionId=35417364
    """
    Assuming all the input files are generated using predict_moiz()
    This function generate's Moiz's blend
    :return:
    """
    res_roberta = belnd_roberta_moiz().rename(columns={'toxic': 'toxic_roberta_xlm'})
    res_bert = blend_bert_moiz().rename(columns={'toxic': 'toxic_bert'})
    res_xlm = blend_xlm_moiz().rename(columns={'toxic': 'toxic_xlm'})

    res = pd.merge(pd.merge(res_roberta, res_bert, on='id'), res_xlm, on='id')

    # The prob blend which scored 9456
    # https://www.kaggle.com/moizsaifee/moiz-submissions-final-blender?scriptVersionId=35344067
    K = 1
    res['toxic_prob'] = ((
                            (9 * (res['toxic_roberta_xlm'] ** (1 / K))) +
                            (2 * (res['toxic_xlm'] ** (1 / K))) +
                            (1 * (res['toxic_bert'] ** (1 / K)))
                    ) / 12) ** K


    # Gen the rank blend which scored 9457
    # https://www.kaggle.com/moizsaifee/moiz-submissions-final-blender?scriptVersionId=35417364
    res['toxic_roberta_xlm'] = res['toxic_roberta_xlm'].rank(pct=True)
    res['toxic_xlm'] = res['toxic_xlm'].rank(pct=True)
    res['toxic_bert'] = res['toxic_bert'].rank(pct=True)
    K = 1
    res['toxic'] = ((
        (9 * (res['toxic_roberta_xlm'] ** (1 / K))) +
        (2 * (res['toxic_xlm'] ** (1 / K))) +
        (1 * (res['toxic_bert'] ** (1 / K)))
    ) / 12) ** K

    return res[['id', 'toxic', 'toxic_prob']]
    
    
def predict_ujjwal():
    os.system('/bin/bash inference.sh')
    
def predict_igor():
    os.system('pip install torchvision > /dev/null')
    os.system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
    os.system('python pytorch-xla-env-setup.py --version 20200420 --apt-packages libomp5 libopenblas-dev ')
    os.system('pip install transformers==2.11.0 > /dev/null')
    os.system('pip install pandarallel > /dev/null')
    os.system('pip install catalyst==20.4.2 > /dev/null')
    
    models_list = [['./Output/Models/Igor/fr/inf_fr_yandex_camembert_large','camembert/camembert-large'],
                   ['./Output/Models/Igor/fr/inf_fr_google_camembert_large','camembert/camembert-large'],
                   ['./Output/Models/Igor/es/inf_es_google_wwm','dccuchile/bert-base-spanish-wwm-cased'],
                   ['./Output/Models/Igor/es/inf_es_yandex_wwm','dccuchile/bert-base-spanish-wwm-cased'],
                   ['./Output/Models/Igor/ru/inf_ru_google_conv','DeepPavlov/rubert-base-cased-conversational'],
                   ['./Output/Models/Igor/ru/inf_ru_yandex_conv','DeepPavlov/rubert-base-cased-conversational'],
                   ['./Output/Models/Igor/tr/inf_tr_google_dbmdz','dbmdz/bert-base-turkish-cased'],
                   ['./Output/Models/Igor/tr/inf_tr_google_savasy','savasy/bert-turkish-text-classification'],
                   ['./Output/Models/Igor/it/inf_it_yandex_xxl','dbmdz/bert-base-italian-xxl-uncased']]
    
    for p in models_list:
        model_file_prefix = p[0]
        backbone = p[1]       
        cmd = f'python "./Code/Igor/predict.py" --in_file="./Input/Igor/test.csv.zip" ' \
            f'--model_file_prefix="{model_file_prefix}" ' \
            f'--backbone="{backbone}" '
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    predict_moiz()
    res = blend_moiz()
    res.to_csv('./Output/Predictions/Moiz/submission_MAS.csv')

    os.chdir('Code/Ujjwal')
    predict_ujjwal()
    
    os.chdir('../../')
    predict_igor()
    
    os.system('python blend.py')

