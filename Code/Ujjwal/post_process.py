import os
import argparse
import pandas as pd 
import numpy as np 
from glob import glob
from functools import reduce

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
args = parser.parse_args()

base = pd.read_csv(args.path + 'source/source_1/test_foreign.csv')['original'].tolist()

datasets = []

for file in glob(args.path + '/step-3/submission*.csv'):
    dataset = pd.read_csv(file)[['id','score']]
    dataset.columns = ['id','toxic']
    dataset['toxic'] = dataset['toxic'] / dataset['toxic'].mean()
    dataset['toxic'] = dataset['toxic'] * 0.21
    dataset['weight'] = base
    datasets.append(dataset)

datasets = reduce(lambda x,y : x.append(y), datasets)

no_tta = datasets.copy()
with_tta = datasets.copy()

no_tta['weight'] = no_tta['weight'].map(lambda x : 0. if x != 1. else 1.)
no_tta['toxic'] = no_tta['toxic'] * no_tta['weight']
no_tta = no_tta.groupby('id')['toxic','weight'].sum().reset_index()
no_tta['toxic'] = no_tta['toxic'] / no_tta['weight']
no_tta = no_tta[['id','toxic']]
no_tta['toxic'] = no_tta['toxic'] / no_tta['toxic'].mean()
no_tta['toxic'] = no_tta['toxic'] * 0.21

with_tta['weight'] = with_tta['weight'].map(lambda x : 5. if x == 1. else 1.)
with_tta['toxic'] = with_tta['toxic'] * with_tta['weight']
with_tta = with_tta.groupby('id')['toxic','weight'].sum().reset_index()
with_tta['toxic'] = with_tta['toxic'] / with_tta['weight']
with_tta = with_tta[['id','toxic']]
with_tta['toxic'] = with_tta['toxic'] / with_tta['toxic'].mean()
with_tta['toxic'] = with_tta['toxic'] * 0.21

no_tta.to_csv(args.path + 'no_tta.csv', index=False)
with_tta.to_csv(args.path + 'with_tta.csv', index=False)