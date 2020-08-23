## Code borrowed from https://www.kaggle.com/riblidezso/train-from-mlm-finetuned-xlm-roberta-large

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import transformers
from transformers import TFRobertaModel, AutoTokenizer
import logging
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


def load_jigsaw_trans(index, langs=['tr','it','es','ru','fr','pt'], 
                      columns=['comment_text', 'toxic']):
    train_6langs=[]
    for i in range(len(langs)):
        fn = FILEPATH2 + 'jigsaw-toxic-comment-train-google-%s-cleaned.csv'%langs[i]
        fn = pd.read_csv(fn)[columns].sample(frac=1., random_state=i).reset_index(drop=True)
        train_6langs.append(downsample(fn, index))
    return train_6langs

def downsample(df, index):
    print(df.toxic.value_counts())
    pos = df.query('toxic==1').reset_index(drop=True)
    neg = df.query('toxic==0').reset_index(drop=True)
    neg = neg[neg.index % 10 == index].reset_index(drop=True)
    print(pos.shape, neg.shape)
    ds_df = pos.append(neg).sample(frac=1.).reset_index(drop=True)
    print(ds_df.shape)
    return ds_df
    
def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(texts, 
        pad_to_max_length=True, 
        max_length=maxlen)
    return np.array(enc_di['input_ids']) 

def create_dist_dataset(X, y=None, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(X)
    if y is not None:
        dataset_y = tf.data.Dataset.from_tensor_slices(y)
        dataset = tf.data.Dataset.zip((dataset, dataset_y))
    if training:
        dataset = dataset.shuffle(len(X)).repeat()
    dataset = dataset.batch(global_batch_size).prefetch(AUTO)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    return dist_dataset
    
def create_model_and_optimizer():
    with strategy.scope():
        transformer_layer = TFRobertaModel.from_pretrained(PRETRAINED_MODEL)                
        model = build_model(transformer_layer)
        optimizer_transformer = Adam(learning_rate=LR_TRANSFORMER)
        optimizer_head = Adam(learning_rate=LR_HEAD)
    return model, optimizer_transformer, optimizer_head


def build_model(transformer):
    inp = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")
    x = transformer(inp)[0][:, 0, :]  
    x = Dropout(DROPOUT)(x)
    out = Dense(1, activation='sigmoid', name='custom_head')(x)
    model = Model(inputs=[inp], outputs=[out])
    return model

def define_losses_and_metrics():
    with strategy.scope():
        loss_object = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE, from_logits=False)
        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            loss = tf.nn.compute_average_loss(
                per_example_loss, global_batch_size = global_batch_size)
            return loss
        train_accuracy_metric = tf.keras.metrics.AUC(name='training_AUC')
    return compute_loss, train_accuracy_metric

def train(train_dist_dataset, val_dist_dataset=None, y_val=None,
          total_steps=2000, validate_every=200):
    best_weights, history = None, []
    step = 0
    for tensor in train_dist_dataset:
        distributed_train_step(tensor) 
        step+=1
        if (step % validate_every == 0):   
            train_metric = train_accuracy_metric.result().numpy()
            print("Step %d, train AUC: %.5f" % (step, train_metric))   
            if val_dist_dataset:
                val_metric = roc_auc_score(y_val, predict(val_dist_dataset))
                print("Step %d,   val AUC: %.5f" %  (step,val_metric))   
                history.append(val_metric)
                if history[-1] == max(history):
                    best_weights = model.get_weights()
            train_accuracy_metric.reset_states()            
        if step  == total_steps:
            break
    model.set_weights(best_weights)

@tf.function
def distributed_train_step(data):
    strategy.experimental_run_v2(train_step, args=(data,))

def train_step(inputs):
    features, labels = inputs
    transformer_trainable_variables = [ v for v in model.trainable_variables 
                                       if (('pooler' not in v.name)  and 
                                           ('custom' not in v.name))]
    head_trainable_variables = [ v for v in model.trainable_variables 
                                if 'custom'  in v.name]
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(features, training=True)
        loss = compute_loss(labels, predictions)
    gradients_transformer = tape.gradient(loss, transformer_trainable_variables)
    gradients_head = tape.gradient(loss, head_trainable_variables)
    del tape
    optimizer_transformer.apply_gradients(zip(gradients_transformer, 
                                              transformer_trainable_variables))
    optimizer_head.apply_gradients(zip(gradients_head, 
                                       head_trainable_variables))
    train_accuracy_metric.update_state(labels, predictions)

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
    parser.add_argument("--fold", type=int)
    parser.add_argument("--pseudo", type=int)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    PATH = args.path
    PRETRAINED_TOKENIZER=  'jplu/tf-xlm-roberta-large'
    MAX_LEN = 192 
    DROPOUT = 0.5
    BATCH_SIZE = 16
    TOTAL_STEPS_STAGE1 = 2000
    VALIDATE_EVERY_STAGE1 = 200
    TOTAL_STEPS_STAGE2 = 200
    VALIDATE_EVERY_STAGE2 = 10
    LR_TRANSFORMER = 5e-6
    LR_HEAD = 1e-3

    PRETRAINED_MODEL = PATH + 'step-2/' + args.mode + '/'
    
    FILEPATH1 = PATH + 'source/source_1/'
    FILEPATH2 = PATH + 'source/source_2/'

    tpu, strategy, global_batch_size = connect_to_TPU()
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    train_df = pd.concat(load_jigsaw_trans(args.fold)).sample(frac=1., random_state=2017)
    train_df = train_df.reset_index(drop=True)
    val_df = pd.read_csv(FILEPATH1 + 'validation.csv')
    if args.pseudo: 
        extra_df = pd.read_csv(FILEPATH1 + 'pseudo_label.csv')[['comment_text', 'toxic']]
        val_df = val_df.append(extra_df).sample(frac=1., random_state=2017).reset_index(drop=True) 
    test_df = pd.read_csv(FILEPATH1 + 'test_foreign.csv')
    sub_df = test_df[['id','lang','weight']].copy()

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TOKENIZER)
    X_train = regular_encode(train_df.comment_text.values, tokenizer, maxlen=MAX_LEN) 
    X_val = regular_encode(val_df.comment_text.values, tokenizer, maxlen=MAX_LEN) 
    X_test = regular_encode(test_df.comment_text.values, tokenizer, maxlen=MAX_LEN) 
    y_train = train_df.toxic.values.reshape(-1,1) 
    y_val = val_df.toxic.values.reshape(-1,1)

    train_dist_dataset = create_dist_dataset(X_train, y_train, True)
    val_dist_dataset   = create_dist_dataset(X_val)
    test_dist_dataset  = create_dist_dataset(X_test)

    model, optimizer_transformer, optimizer_head = create_model_and_optimizer()
    model.summary()
    compute_loss, train_accuracy_metric = define_losses_and_metrics()

    train(train_dist_dataset, val_dist_dataset, y_val, TOTAL_STEPS_STAGE1, VALIDATE_EVERY_STAGE1)
    optimizer_head.learning_rate.assign(1e-4)
    X_train, X_val, y_train, y_val = train_test_split(X_val, y_val, test_size = 0.1)
    train_dist_dataset = create_dist_dataset(X_train, y_train, training=True)
    val_dist_dataset = create_dist_dataset(X_val, y_val)

    train(train_dist_dataset, val_dist_dataset, y_val, total_steps=TOTAL_STEPS_STAGE2, validate_every=VALIDATE_EVERY_STAGE2)

    sub_df['toxic'] = predict(test_dist_dataset)[:,0]
    sub_df.to_csv('submission.csv', index=False)

    model.save(args.path + '/step_3/model_{}.h5'.format(args.out))


