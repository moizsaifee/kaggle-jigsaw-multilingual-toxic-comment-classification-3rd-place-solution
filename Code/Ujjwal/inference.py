import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import transformers
from transformers import TFRobertaModel, AutoTokenizer
AUTO = tf.data.experimental.AUTOTUNE

def connect_to_TPU():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()
    global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync
    return tpu, strategy, global_batch_size

def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(texts, 
        return_token_type_ids=False, 
        pad_to_max_length=True, 
        max_length=maxlen)
    return np.array(enc_di['input_ids']) 

def create_dist_dataset(X):
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.batch(global_batch_size).prefetch(AUTO)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    return dist_dataset

def create_model_and_optimizer():
    with strategy.scope():
        transformer_layer = TFRobertaModel.from_pretrained(PRETRAINED_TOKENIZER)                
        model = build_model(transformer_layer)
    return model

def build_model(transformer):
    inp = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")
    x = transformer(inp)[0][:, 0, :]  
    x = Dropout(DROPOUT)(x)
    out = Dense(1, activation='sigmoid', name='custom_head')(x)
    model = Model(inputs=[inp], outputs=[out])
    return model

def predict(dataset):  
    predictions = []
    for tensor in dataset:
        predictions.append(distributed_prediction_step(tensor))
    predictions = np.vstack(list(map(np.vstack,predictions)))
    return predictions

@tf.function
def distributed_prediction_step(data):
    predictions = strategy.experimental_run_v2(prediction_step, args=(data,))
    return strategy.experimental_local_results(predictions)

def prediction_step(inputs):
    features = inputs
    predictions = model(features, training=False)
    return predictions

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    PATH = args.path
    PRETRAINED_TOKENIZER= 'jplu/tf-xlm-roberta-large'
    MAX_LEN = 192 
    DROPOUT = 0.5
    BATCH_SIZE = 16
    tpu, strategy, global_batch_size = connect_to_TPU()
    score = pd.read_csv(PATH + 'source/source_1/test_foreign.csv')
    submit = score[['id','lang','original']].copy()
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TOKENIZER)
    X_test = regular_encode(score.comment_text.values, tokenizer, maxlen=MAX_LEN) 
    test_dist_dataset  = create_dist_dataset(X_test)
    model = create_model_and_optimizer()
    model.summary()
    model.load_weights(PATH + '/step-3/model_{}.h5'.format(args.mode))
    submit['score'] = predict(test_dist_dataset)
    print(submit)
    submit.to_csv(PATH + '/step-3/submission_{}.csv'.format(args.mode), index=False)

