"""
    This script takes the following thing as input:
        1) An Input file to be scored
        2) BACKBONE used for tokenizer
        3) model checkpoints prefix
    and produces:
        1) Scored input file
        
    Parameters:
        - backbone: Bert based model from huggingface transformers. 'camembert/camembert-large' for example
        - model_file_prefix: prefix of checkpoints name of saved model
        - in_file: The name of (not-processed) input file to be scored. In our case it's test.csv.zip from competition dataset
	Output file = {model_file_prefix}.probs.csv
"""

import numpy as np
import pandas as pd
import os
os.environ['XLA_USE_BF16'] = "1"
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import sklearn
import time
import random
from datetime import datetime
from tqdm import tqdm
tqdm.pandas()
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
import gc
import re
import nltk
nltk.download('punkt')
from nltk import sent_tokenize
from pandarallel import pandarallel
import argparse
pandarallel.initialize(nb_workers=4, progress_bar=False)


parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str)
parser.add_argument('--model_file_prefix', type=str)
parser.add_argument('--in_file', type=str)
parsed_args = parser.parse_args()


SEED = 42
MAX_LENGTH = 224
FILE_NAME = parsed_args.model_file_prefix
BACKBONE_PATH = parsed_args.backbone

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

from nltk import sent_tokenize
from random import shuffle
import random
import albumentations
from albumentations.core.transforms_interface import DualTransform, BasicTransform


LANGS = {
    'en': 'english',
    'it': 'italian', 
    'fr': 'french', 
    'es': 'spanish',
    'tr': 'turkish', 
    'ru': 'russian',
    'pt': 'portuguese'
}

def get_sentences(text, lang='en'):
    return sent_tokenize(text, LANGS.get(lang, 'english'))

def exclude_duplicate_sentences(text, lang='en'):
    sentences = []
    for sentence in get_sentences(text, lang):
        sentence = sentence.strip()
        if sentence not in sentences:
            sentences.append(sentence)
    return ' '.join(sentences)

def clean_text(text, lang='en'):
    text = str(text)
    text = re.sub(r'[0-9"]', '', text)
    text = re.sub(r'#[\S]+\b', '', text)
    text = re.sub(r'@[\S]+\b', '', text)
    text = re.sub(r'https?\S+', '', text)
    text = re.sub(r'\s+', ' ', text)   
    text = exclude_duplicate_sentences(text, lang)
    return text.strip()

tokenizer = AutoTokenizer.from_pretrained(BACKBONE_PATH)
class DatasetRetriever(Dataset):

    def __init__(self, df):
        self.comment_texts = df['comment_text'].values
        self.ids = df['id'].values
        

    def get_tokens(self, text):
        encoded = tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=MAX_LENGTH, 
            pad_to_max_length=True
        )
        return encoded['input_ids'], encoded['attention_mask']

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        text = self.comment_texts[idx]
    
        tokens, attention_mask = self.get_tokens(text)
        tokens, attention_mask = torch.tensor(tokens), torch.tensor(attention_mask)

        return self.ids[idx], tokens, attention_mask

df_test = pd.read_csv(parsed_args.in_file)
df_test['comment_text'] = df_test.parallel_apply(lambda x: clean_text(x['content'], x['lang']), axis=1)
df_test = df_test.drop(columns=['content'])

test_dataset = DatasetRetriever(df_test)

class ToxicSimpleNNModel(nn.Module):

    def __init__(self):
        super(ToxicSimpleNNModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(BACKBONE_PATH)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(
            in_features=self.backbone.pooler.dense.out_features*2,
            out_features=2,
        )

    def forward(self, input_ids, attention_masks):
        bs, seq_length = input_ids.shape
        seq_x, _ = self.backbone(input_ids=input_ids, attention_mask=attention_masks)
        apool = torch.mean(seq_x, 1)
        mpool, _ = torch.max(seq_x, 1)
        x = torch.cat((apool, mpool), 1)
        x = self.dropout(x)
        return self.linear(x)

import warnings

warnings.filterwarnings("ignore")

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


class MultiTPUPredictor:
    
    def __init__(self, model, device):



        self.model = model
        self.device = device

        xm.master_print(f'Model prepared. Device is {self.device}')


    def run_inference(self, test_loader, e, verbose=True, verbose_step=50):
        self.model.eval()
        result = {'id': [], 'toxic': []}
        t = time.time()
        for step, (ids, inputs, attention_masks) in enumerate(test_loader):
            if verbose:
                if step % 50 == 0:
                    xm.master_print(f'Prediction Step {step}, time: {(time.time() - t):.5f}')

            with torch.no_grad():
                inputs = inputs.to(self.device, dtype=torch.long) 
                attention_masks = attention_masks.to(self.device, dtype=torch.long)
                outputs = self.model(inputs, attention_masks)
                toxics = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()[:,1]

            result['id'].extend(ids.numpy())
            result['toxic'].extend(toxics)

        result = pd.DataFrame(result)
        node_count = len(glob('node_submissions/*.csv'))
        result.to_csv(f'node_submissions/submission_{e}_{node_count}_{datetime.utcnow().microsecond}.csv', index=False)

def _mp_fn(rank, flags):
    device = xm.xla_device()
    model = net.to(device)

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        sampler=test_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=1
    )

    fitter = MultiTPUPredictor(model=model, device=device)
    fitter.run_inference(test_loader,E)

if not os.path.exists('node_submissions'):
    os.makedirs('node_submissions')
else:
    files = glob('node_submissions/*')
    for f in files:
        os.remove(f)
net = ToxicSimpleNNModel()
checkpoints = glob(f'{FILE_NAME}*bin')
for n,cp in enumerate(checkpoints):
    E = n + 1
    print(f'cp number {E}')
    checkpoint = torch.load(cp, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint);
    checkpoint = None
    del checkpoint
    FLAGS={}
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')

submission = pd.concat([pd.read_csv(path) for path in glob('node_submissions/*.csv')]).groupby('id').mean()
submission.to_csv(parsed_args.model_file_prefix + ".prob.csv")
