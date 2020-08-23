import argparse
import pandas as pd
import numpy as np 
from transformers import XLMRobertaTokenizer

TOKENIZER = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

def regular_encode(texts):
    encode = TOKENIZER.batch_encode_plus(
        texts, 
        pad_to_max_length=True,
        max_length=128
    )
    return np.array(encode['input_ids'])

def save_encodings():
	data1 = pd.read_csv(PATH + '/source/source_1/train_english.csv')[['comment_text']] 
	data2 = pd.read_csv(PATH + '/source/source_1/train_foreign.csv')[['comment_text']] 
	data3 = pd.read_csv(PATH + '/source/source_1/subtitle.csv')[['comment_text']] 
	data = data1.append(data2).append(data3)
	data = regular_encode(data['comment_text'].values.tolist())
	print('train_encode.npz:', data.shape)
	np.savez(PATH + '/step_1/input/train_encode.npz', arr_0=data)

	data1 = pd.read_csv(PATH + '/source/source_1/valid_foreign.csv')[['comment_text']] 
	data2 = pd.read_csv(PATH + '/source/source_1/valid_english.csv')[['comment_text']] 
	data = data1.append(data2)
	print('valid_encode.npz:', data.shape)
	np.savez(PATH + '/step_1/input/valid_encode.npz', arr_0=data)

	data1 = pd.read_csv(PATH + '/source/source_1/test_foreign.csv')[['comment_text']] 
	data2 = pd.read_csv(PATH + '/source/source_1/test_english.csv')[['comment_text']] 
	data = data1.append(data2)
	print('test_encode.npz:', data.shape)
	np.savez(PATH + '/step_1/input/test_encode.npz', arr_0=data)

	data1 = pd.read_csv(PATH + '/source/source_1/train_english.csv') 
	data1 = data1[data1['source'] == '2020-train'][['comment_text']]
	data2 = pd.read_csv(PATH + '/source/source_1/train_foreign.csv') 
	data2 = data2[data2['source'] == '2020-train'][['comment_text']]
	data3 = pd.read_csv(PATH + '/source/source_1/valid_foreign.csv')
	data3 = data3[data3['original'] == 1][['comment_text']]
	data4 = pd.read_csv(PATH + '/source/source_1/test_foreign.csv')
	data4 = data4[data4['original'] == 1][['comment_text']]
	data = data1.append(data2).append(data3).append(data4)
	print('data_subset.npz:', data.shape)
	np.savez(PATH + '/step_1/input/data_subset.npz', arr_0=data)

	data1 = pd.read_csv(PATH + '/source/source_1/valid_foreign.csv')[['comment_text']]
	data2 = pd.read_csv(PATH + '/source/source_1/test_foreign.csv')[['comment_text']]
	data3 = pd.read_csv(PATH + '/source/source_1/valid_english.csv')[['comment_text']]
	data4 = pd.read_csv(PATH + '/source/source_1/test_english.csv')[['comment_text']]
	data = data1.append(data2).append(data3).append(data4)
	print('data.npz:', data.shape)
	np.savez(PATH + '/step_0/input/data.npz', arr_0=data)
	return None

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    PATH = args.path
	save_encodings()
