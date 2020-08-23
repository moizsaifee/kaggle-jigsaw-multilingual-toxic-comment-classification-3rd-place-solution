import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm.autonotebook import tqdm
from sklearn.model_selection import StratifiedKFold
import re
import time

RANDOM_STATE = 5353
np.random.seed(RANDOM_STATE)
# Import
from pandarallel import pandarallel

# Initialization
pandarallel.initialize()


def clean_text(text):
    text = str(text)
    text = re.sub(r'[0-9"]', '', text)
    text = re.sub(r'#[\S]+\b', '', text)
    text = re.sub(r'@[\S]+\b', '', text)
    text = re.sub(r'https?\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def encode(texts, tokenizer, pad, max_len):
    enc_di = tokenizer.batch_encode_plus(
        texts,
        return_attention_masks=False,
        return_token_type_ids=False,
        pad_to_max_length=pad,
        max_length=max_len
    )
    return [np.array(x) for x in enc_di['input_ids']]


def split_encode(inp, window_length, max_len):
    """
    :param inp:
    :param window_length:
    :param max_len:
    :return:
    """
    out_all = []
    num_unchanged_ids = 0
    num_split_ids = 0
    num_split_rows = 0
    input_id_loc = inp.columns.get_loc('input_ids')

    for row in inp.itertuples():
        if len(row.input_ids) < max_len:
            input_ids_current = list(row.input_ids)
            num_unchanged_ids += 1
            pad_length = max_len - len(row.input_ids)
            if pad_length > 0:
                input_ids_current += [1] * pad_length
            row_ref = list(row)[1:]
            row_ref[input_id_loc] = np.array(input_ids_current)
            out_all.append(row_ref)
        else:
            input_ids_content = list(row.input_ids)[1:-1]
            num_split_ids += 1
            start = 0
            while True:
                if len(input_ids_content[start:]) <= max_len - 2:
                    # This is the last row
                    input_ids_current = [0] + input_ids_content[start:start + max_len - 2] + [2]
                    pad_length = max_len - len(input_ids_current)
                    if pad_length > 0:
                        input_ids_current += [1] * pad_length
                    row_ref = list(row)[1:]
                    row_ref[input_id_loc] = np.array(input_ids_current)
                    out_all.append(row_ref)
                    num_split_rows += 1
                    break
                else:
                    input_ids_current = [0] + input_ids_content[start:start + max_len - 2] + [2]
                    start += window_length
                    # No padding should be needed
                    row_ref = list(row)[1:]
                    row_ref[input_id_loc] = np.array(input_ids_current)
                    out_all.append(row_ref)
                    num_split_rows += 1

    out_df = pd.DataFrame(out_all)
    out_df.columns = inp.columns
    print(f'SUMMARY: unchanged: {num_unchanged_ids} split: {num_split_ids} rows: {num_split_rows} out_df{out_df.shape}')
    return out_df


def dump_chunk(train, max_len, tokenizer, out_dir, out_prefix, out_suffix, long_comment_action):
    train = train.copy().sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Clean
    print('Started Cleaning: ', time.ctime())
    # Temp - commenting off cleaning for template messages
    train['comment_text'] = train.parallel_apply(lambda x: clean_text(x['comment_text']), axis=1)
    print('End Cleaning', time.ctime())

    original_shape = train.shape
    if 'strata' in train.columns:
        del train['strata']

    if long_comment_action == 'split':
        train['input_ids'] = encode(train['comment_text'].values, tokenizer, False, 1000)
        del train['comment_text']
        train = split_encode(train, 100, max_len)

    elif long_comment_action == 'drop':
        train['input_ids'] = encode(train['comment_text'].values, tokenizer, True, max_len)
        del train['comment_text']
        keep_flag = train['input_ids'].map(lambda x: True if x[-1] == 1 else False)
        train = train[keep_flag].reset_index(drop=True)

    elif long_comment_action == 'ignore':
        train['input_ids'] = encode(train['comment_text'].values, tokenizer, True, max_len)
        del train['comment_text']

    else:
        raise ValueError

    print(f'{out_prefix}_{out_suffix} Original: {original_shape}: Final: {train.shape}')
    train.to_pickle(f'{out_dir}/{out_prefix}_{out_suffix}.pkl')

def process_file(
        max_len,
        model_name,
        train,
        file_type : str,
        out_dir,
        out_prefix,
        should_chunk: bool, # whether should break input file into smaller files
        chunk_size : int,
        long_comment_action: str,
        max_chunk = None
        ):
    """
    1) Read the full file
    2) Shuffle
    3) Dump into Chunks of 50K samples

    :param inp_file:
    :return:
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # train = read_file(f'{in_dir}/{inp_file}', file_type)

    if file_type == 'train':
        print(train['toxic_float'].value_counts())
        print(train['toxic_float'].value_counts(normalize=True))

    train = train.dropna().reset_index(drop=True)
    print(f'# Rows after dropping NA: {train.shape}')

    if should_chunk and file_type != 'test':
        k_fold = train.shape[0] // chunk_size
        if k_fold > 2:
            train['strata'] = train['lang'] + train['toxic_float'].astype('str')
            print(train['strata'].value_counts())
            print(train['strata'].value_counts(normalize=True))
            sss = StratifiedKFold(n_splits=k_fold, random_state=RANDOM_STATE)
            for i, (train_index, test_index) in tqdm(enumerate(sss.split(np.zeros(train.shape[0]), train['strata']))):
                if (max_chunk is not None) and (i >= max_chunk):
                    print('Max Chunks Processed, Exiting')
                    break
                dump_chunk(train.iloc[test_index], max_len, tokenizer, out_dir, out_prefix, f'p{i}', long_comment_action)

        else:
            print('Not Enough observation in inp to break into chunks')
            dump_chunk(train, max_len, tokenizer, out_dir, out_prefix, 'p0', long_comment_action)
    else:
        dump_chunk(train, max_len, tokenizer, out_dir, out_prefix, 'p0', long_comment_action)