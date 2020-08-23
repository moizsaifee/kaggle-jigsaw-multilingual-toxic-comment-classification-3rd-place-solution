## Code borrowed from https://www.kaggle.com/riblidezso/finetune-xlm-roberta-on-jigsaw-test-data-with-mlm

import argparse
from ast import literal_eval
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import transformers
from transformers import TFAutoModelWithLMHead, AutoTokenizer
import logging
logging.getLogger().setLevel(logging.NOTSET)
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

def prepare_mlm_input_and_labels(X):
    inp_mask = np.random.rand(*X.shape)<0.15 
    inp_mask[X<=2] = False
    labels =  -1 * np.ones(X.shape, dtype=int)
    labels[inp_mask] = X[inp_mask]
    X_mlm = np.copy(X)
    inp_mask_2mask = inp_mask  & (np.random.rand(*X.shape)<0.90)
    X_mlm[inp_mask_2mask] = 250001  # mask token is the last in the dict
    inp_mask_2random = inp_mask_2mask  & (np.random.rand(*X.shape) < 1/9)
    X_mlm[inp_mask_2random] = np.random.randint(3, 250001, inp_mask_2random.sum())    
    return X_mlm, labels

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
        
def create_mlm_model_and_optimizer():
    with strategy.scope():
        model = TFAutoModelWithLMHead.from_pretrained(PRETRAINED_MODEL)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    return model, optimizer

def define_mlm_loss_and_metrics():
    with strategy.scope():
        mlm_loss_object = masked_sparse_categorical_crossentropy
        def compute_mlm_loss(labels, predictions):
            per_example_loss = mlm_loss_object(labels, predictions)
            loss = tf.nn.compute_average_loss(
                per_example_loss, global_batch_size = global_batch_size)
            return loss
        train_mlm_loss_metric = tf.keras.metrics.Mean()
    return compute_mlm_loss, train_mlm_loss_metric

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked, from_logits=True)
    return loss

def train_mlm(train_dist_dataset, total_steps=2000, evaluate_every=200):
    step = 0
    for tensor in train_dist_dataset:
        distributed_mlm_train_step(tensor) 
        step+=1
        if (step % evaluate_every == 0):   
            train_metric = train_mlm_loss_metric.result().numpy()
            print("Step %d, train loss: %.2f" % (step, train_metric))     
            train_mlm_loss_metric.reset_states()            
        if step  == total_steps:
            break

@tf.function
def distributed_mlm_train_step(data):
    strategy.experimental_run_v2(mlm_train_step, args=(data,))

@tf.function
def mlm_train_step(inputs):
    features, labels = inputs
    with tf.GradientTape() as tape:
        predictions = mlm_model(features, training=True)[0]
        loss = compute_mlm_loss(labels, predictions)
    gradients = tape.gradient(loss, mlm_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mlm_model.trainable_variables))
    train_mlm_loss_metric.update_state(loss)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    PATH = args.path
    MAX_LEN = 128
    BATCH_SIZE = 16
    TOTAL_STEPS = 10000 
    EVALUATE_EVERY = 200
    LR =  1e-5
    PRETRAINED_MODEL = 'jplu/tf-xlm-roberta-large'

    np.random.seed(2017)
    tpu, strategy, global_batch_size = connect_to_TPU()
    print("REPLICAS: ", strategy.num_replicas_in_sync)

    if args.mode == 'version1':
        X_trn = np.load(PATH + 'step-1/train_encode.npz')['arr_0']
        X_val = np.load(PATH + 'step-1/valid_encode.npz')['arr_0']
        X_tst = np.load(PATH + 'step-1/test_encode.npz')['arr_0']
        X_train_mlm = np.vstack([X_trn, X_val, X_tst])
        np.random.shuffle(X_train_mlm)
        X_train_mlm = X_train_mlm[:1000000,:]
    if args.mode == 'version2':
        X_trn = np.load(PATH + 'step-1/data.npz')['arr_0']
        X_train_mlm = X_trn
        np.random.shuffle(X_train_mlm)
        X_train_mlm = X_train_mlm[:1000000,:]
    if args.mode == 'version3':
        X_trn = np.load(PATH + 'step-1/data_subset.npz')['arr_0']
        X_train_mlm = X_trn
        np.random.shuffle(X_train_mlm)

    X_train_mlm, y_train_mlm = prepare_mlm_input_and_labels(X_train_mlm)
    train_dist_dataset = create_dist_dataset(X_train_mlm, y_train_mlm, True)
    mlm_model, optimizer = create_mlm_model_and_optimizer()
    mlm_model.summary()
    compute_mlm_loss, train_mlm_loss_metric = define_mlm_loss_and_metrics()
    train_mlm(train_dist_dataset, TOTAL_STEPS, EVALUATE_EVERY)
    mlm_model.save_pretrained(PATH + 'step-2/' + args.mode + '/')

