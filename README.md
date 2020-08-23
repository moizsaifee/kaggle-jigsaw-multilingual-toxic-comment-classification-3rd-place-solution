Hello!

Here we provide a high level overview and various underlying details of the 3rd place winning solution in the Kaggle Jigsaw Multilingual Toxic Comment classification. 

Please note that while we are sharing the complete code and blending scripts etc, we can't share the input data / cahched models etc due to file size limit on Github.

# Solution Summary
![Solution Summary](https://github.com/moizsaifee/kaggle-jigsaw-multilingual-toxic-comment-classification-3rd-place-solution/blob/master/img/1.png)

# Additional High level details 
![Solution Summary](https://github.com/moizsaifee/kaggle-jigsaw-multilingual-toxic-comment-classification-3rd-place-solution/blob/master/img/2.png)
![Solution Summary](https://github.com/moizsaifee/kaggle-jigsaw-multilingual-toxic-comment-classification-3rd-place-solution/blob/master/img/3.png)

# Code / Implementation 

## MODEL BUILD: There are three options to produce the solution.
1) very fast prediction
    a) runs in a few minutes
    b) uses precomputed neural network predictions
2) ordinary prediction (use models from Output/Models/Igor/,Output/Models/Moiz/,Input/Ujjwal/Data/step-[2-3]/*h5)
    a) expect this to run for 5-6 hours
    b) uses binary model files
3) retrain models
    a) expect this to run about 1-2 days
    b) trains all models from scratch
    c) follow this with (2) to produce entire solution from scratch

command to run each build is below
1) very fast prediction (overwrites Output/Predictions/submission.csv)
python ./blend.py

2) ordinary prediction (overwrites Output/Predictions/submission.csv, Output/Predictions/Moiz/*csv, Output/Predictions/Ujjwal/*csv, Output/Models/Igor/{lang}/*probs.csv )
python ./inference.py

3) retrain models (overwrites models in Output/Models/Igor_dev/,Output/Models/Moiz_dev/,Input/Ujjwal/Data/step-[2-3]/*h5)
python ./train.py



## CONTENTS
Input/Igor/test.csv.zip       :original kaggle test data
Input/Igor/validation.csv.zip         : original kaggle validation data

Input/Igor/train_data_{lang}_google.csv.zip : translated train data via google translater (https://www.kaggle.com/miklgr500/jigsaw-train-multilingual-coments-google-api)

Input/Igor/train_data_{lang}_yandex.csv.zip : translated train data via yandex translater (https://www.kaggle.com/ma7555/jigsaw-train-translated-yandex-api)

Input/Igor/633287_1126366_compressed_open-subtitles-synthesic.csv.zip : pseudolabeling open_subtitles (https://www.kaggle.com/shonenkov/open-subtitles-toxic-pseudo-labeling)


Input/Common/Raw/ : original kaggle datasets + pseudolabeling open_subtitles and pseudolabeling test_data
Input/Common/Processed_Ujjwal/ : translated datasets


Input/Moiz/train/ : data produced by ./prepare_data_train.py
Input/Moiz/test/ : data produced by ./prepare_data_inference.py


Output/Models/Igor/{lang}/*bin : PyTorch checkpoints of mono-lingual models 
Output/Models/Moiz/*h5 : TF checkpoints of MAS models
Input/Ujjwal/Data/step-[2-3]/*h5 : TF checkpoints of MLM models

Output/Models/Igor/{lang}/*probs.csv : predictions of mono-lingual models
Output/Predictions/Moiz/*csv : predictions of MAS models 
Input/Ujjwal/Data/*tta.csv : predictions of MLM models 

Output/Predictions/submission.csv : final submission file



## HARDWARE: (The following specs were used to create the original solution)
v3-128 TPU - need for TF training (All TF models were trained via Kaggle). It's important that instance has 16Gb memory per core (128 totally)
64Gb memory - need for PyTorch training (All PyTorch models were trained via Google Colab Pro which has more memory than Kaggle instance but less TPU memory (8 vs 16))
Access to internet for downloading packages


## SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.6.9


WARNING! Do no install pytorch-xla-env-setup.py before starting TF code. In this case there is an incompatibility in using TPU via TF and via PyTorch in the same instance runtime. The valid sequence of running (including install packages) is in ./train.py and ./inference.py.


## DATA SETUP


## DATA PROCESSING
### The train/predict code will also call this script if it has not already been run on the relevant data.
python ./prepare_data_train.py
python ./prepare_data_inference.py






############## Ujjwal model description
The following code repository produces the the submission file for MLM part.

The code is borrowed from @riblidezso's following notebooks:

- [Pre-training Roberta-XLM](https://www.kaggle.com/riblidezso/finetune-xlm-roberta-on-jigsaw-test-data-with-mlm)
- [Supervised Training Roberta-XLM](https://www.kaggle.com/riblidezso/train-from-mlm-finetuned-xlm-roberta-large)

These codes assume TPU access.


## Overview

### Input Data:

There are two sources of input data:

- source_1
	- train_english: given english dataset for toxic comments
	- train_foreign: train_english dataset translated to foreign dataset
	- valid_english: validation data translated to english
	- valid_foreign: original validation dataset
	- test_english: test dataset translated to english
	- test_foreign: original test dataset
	- subtitle: open subtitle dataset 
	- pseudo_label: given test dataset pseudo-labeled based on our model prediction scores

- source_2:
	- [Public Dataset](https://www.kaggle.com/miklgr500/jigsaw-train-multilingual-coments-google-api)

We used three different input pipelines to pre-train XLM models. We translated each record to various languages (en, es, fr, tr, ru, it, pt) to obtain more data for pre-training the model.

- Version 1: Translated train, valid and test
- Version 2: Translated train and open subtitle dataset 
- Version 3: Translated validation and test

### Step - 1: Data Processing

Code: encode.py
Input: source/source_1 files
Output: encoded npz arrays step_1

We encoded the text in CSV files to create numpy arrays with numerical encodings. We did this to reduce the TPU runtime of the notebooks. The encoded arrays can be found in this [Kaggle Dataset](https://www.kaggle.com/brightertiger/jigsawencode)

### Step - 2: Pre-training

Code: pretrain_xlm.py
Input: encoded npz arrays step_1
Output: xlm-model weights step_2/version*

We used the three input versions to pre-train three XLM-Roberta models using masked language modeling. 

The Kaggle Scripts corresponding to three versions are:
- [Version 1](https://www.kaggle.com/brightertiger/finetune-xlm-roberta-on-jigsaw-test-data-with-mlm?scriptVersionId=35754034)
- [Version 2](https://www.kaggle.com/brightertiger/finetune-xlm-roberta-on-jigsaw-test-data-with-mlm?scriptVersionId=35762322)
- [Version 3](https://www.kaggle.com/brightertiger/finetune-xlm-roberta-on-jigsaw-test-data-with-mlm?scriptVersionId=35904862)

These three versions output three XLM-Roberta models that are used for supervised training in the next step. The models are saved here.

- [Version 1](https://www.kaggle.com/brightertiger/mlmv1)
- [Version 2](https://www.kaggle.com/brightertiger/mlmv2)
- [Version 3](https://www.kaggle.com/brightertiger/mlmv3)

### Step - 3: Fine-tuning

Code: finetune_xlm.py
Input: source/source_2 files, step-2/input models, fold-idx
Output: step-3 model weights

The three models from previous version are fine-tuned using task labels in this step. The models train best when downsampled 1:1 ratio of toxic and non-toxic labels. To ensure this, each model is fine-tuning task in triggered ~10 times with each a different subset for non-toxic labels. To add more diversity to the training pipeline, in half of the runs pseudo-labels (generated from our predictions) were added to the validation dataset. 

The Kaggle scripts for this version can be found at:

- [Version 1](https://www.kaggle.com/brightertiger/mlm-v1-code)
- [Version 2](https://www.kaggle.com/brightertiger/mlm-v2-code)
- [Version 3](https://www.kaggle.com/brightertiger/mlm-v3-code)

### Step - 4: Inference

Code: inference.py
Input: step-3 model weights
Output: step-3 score files

These codes can be used for running inference based on the models trained in Step-2. For running scoring on new file, replace it with the one located in /Input/Ujjwal/Data/source/source_1/test_foreign.csv.

### Step - 5: Post-Processing

Code: post-process.py
Input: step-3 score files
Output: with_tta.csv, without_tta.csv

The final output is blended is generated by averaging the two versions with and without test-time augmentation (TTA).

- Without TTA: use only those records present in original test. Give zero weight to everything else.
- With TTA: use records present in original file (weight=5.) and records obtained from translation (weight=1.)

These files are avialable in [output folder](Output/Predictions/Ujjwal). A Simple average of these files scores 0.9460 and 0.9446 on pulic and private leaderboards respectively. These files are then combined with the scores from my other team mates in the final blend.






