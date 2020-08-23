from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
from transformers import TFAutoModel
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, GlobalMaxPool1D, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import transformers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import gc

"""
This File contains all the function to train / score a model
"""

def set_seed():
    seed = 5353
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def init_logger():
    logger = getLogger()
    logger.setLevel(INFO)
    # if not logger.hasHandlers():
    handler = StreamHandler()
    format_str = '%(asctime)s: %(funcName)20s() -- %(message)s'
    handler.setFormatter(Formatter(format_str))
    logger.addHandler(handler)
    return logger


def plotit(train_history):
    # summarize history for accuracy
    plt.plot(train_history.history['accuracy'])
    plt.plot(train_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)


def build_model(model_name, max_len, init_lr=1e-5, label_smoothing=0):
    if model_name == 'jplu/tf-xlm-roberta-large':
        config = transformers.RobertaConfig.from_pretrained(model_name)
    elif model_name == 'xlm-mlm-100-1280':
        config = transformers.XLMConfig.from_pretrained(model_name)
    elif model_name == 'bert-base-multilingual-cased':
        config = transformers.BertConfig.from_pretrained(model_name)
    else:
        raise NotImplementedError
    config.output_hidden_states = True
    transformer_layer = TFAutoModel.from_pretrained(model_name, config=config)
    input_word_ids = Input(shape=(max_len,), dtype=tf.int64, name="input_word_ids")

    if model_name == 'xlm-mlm-100-1280':
        _, outputs = transformer_layer(input_word_ids)
    else:
        _, _, outputs = transformer_layer(input_word_ids)

    outputs_sub = Concatenate()([outputs[-1], outputs[-2], outputs[-3], outputs[-4]])

    avg_pool = GlobalAveragePooling1D()(outputs_sub)
    max_pool = GlobalMaxPool1D()(outputs_sub)

    pooled_output = Concatenate()([max_pool, avg_pool])

    pooled_output = Dropout(0.5)(pooled_output)
    out = Dense(300, activation=lrelu)(pooled_output)
    out = Dropout(0.5)(out)
    out = Dense(1, activation='sigmoid')(out)

    model = Model(inputs=input_word_ids, outputs=out)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    model.compile(Adam(lr=init_lr), loss=loss, metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model


def get_strategy():
    # TPU Config
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print(f'Running on TPU {tpu.master()}')
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        print('NOT running on TPU ')
        strategy = tf.distribute.get_strategy()
    return strategy


class Solver:
    def __init__(self, logger, config):
        self.config = config
        self.logger = logger

        self.train_size = None
        self.max_epochs = None
        self.prefetch_policy = tf.data.experimental.AUTOTUNE

        # Check whether TPU available or need to run on GPU / CPU etc
        # Update the Batch size if running in a distributed fashion
        self.strategy = get_strategy()
        self.config['global_batch_size'] = self.config["batch_size"] * self.strategy.num_replicas_in_sync
        self.logger.info(f'Batch Size: {self.config["batch_size"]} '
                         f'Global Batch Size: {self.config["global_batch_size"]}')

    def get_data_loaders(self, train_files: list, val_file, class_balance, class_ratio, min_thresh, max_thresh):

        train_data = []
        for index, train_file in enumerate(train_files):
            self.logger.info(train_file)
            temp = pd.read_pickle(train_file)
            self.logger.info(f'# Rows - raw: {temp.shape}')
            temp = temp[(temp['toxic_float'] < min_thresh) | (max_thresh <= temp['toxic_float'])]
            self.logger.info(f'# Rows - thresholding: {temp.shape}')
            temp['toxic'] = temp['toxic_float'].map(lambda x: 1 if x >= max_thresh else 0)
            del temp['toxic_float']
            self.logger.info(temp['toxic'].value_counts(normalize=True))
            if class_balance:
                num_pos = temp[temp['toxic'] == 1].shape[0]
                num_neg = temp[temp['toxic'] == 0].shape[0]
                to_sel_neg = np.minimum(num_pos * class_ratio, num_neg)
                temp = pd.concat(
                    [temp[temp['toxic'] == 1], temp[temp['toxic'] == 0].sample(to_sel_neg, random_state=5353)],
                    axis=0).sample(frac=1, random_state=5353).reset_index(drop=True)
            train_data.append(temp)
        train_data = pd.concat(train_data, axis=0).reset_index(drop=True)
        self.logger.info(f'# Rows Train Original (concatenated): {train_data.shape}')
        gc.collect()

        # Ensure that toxic column has just two unique values
        assert train_data['toxic'].nunique() == 2

        val_data = pd.read_pickle(val_file)
        val_data = val_data.rename(columns={'toxic_float': 'toxic'})
        if self.config['dev_mode']:
            train_data = train_data.sample(np.minimum(256, train_data.shape[0])).reset_index(drop=True)
            val_data = val_data.sample(np.minimum(256, val_data.shape[0])).reset_index(drop=True)
            self.max_epochs = 1

        self.logger.info(f'# Rows Train: {train_data.shape}')
        self.logger.info(f'# Rows Val: {val_data.shape}')

        train_dataset = (
            tf.data.Dataset
                .from_tensor_slices(
                (np.array(train_data['input_ids'].values.tolist()), train_data['toxic'].values))
                .repeat()
                .shuffle(train_data.shape[0])
                .batch(self.config['global_batch_size'])
                .prefetch(self.prefetch_policy)
        )

        valid_dataset = (
            tf.data.Dataset
                .from_tensor_slices(
                (np.array(val_data['input_ids'].values.tolist()), val_data['toxic'].values))
                .batch(self.config['global_batch_size'])
                .cache()
                .prefetch(self.prefetch_policy)
        )
        self.train_size = train_data.shape[0]
        return train_dataset, valid_dataset

    def train(self, train_files: list, val_file, model_save_file,
              max_epochs, patience, init_lr,
              class_balance, class_ratio, label_smoothing,
              min_thresh, max_thresh,
              epoch_split_factor=1, model_resume_file=None):
        # Load the training and validation data and create Loaders
        self.max_epochs = max_epochs
        train_dataset, valid_dataset = self.get_data_loaders(
            train_files, val_file, class_balance, class_ratio, min_thresh, max_thresh)
        # Create a callback that saves the model's weights

        with self.strategy.scope():
            model = build_model(self.config['model_name'], max_len=self.config['max_len'], init_lr=init_lr,
                                label_smoothing=label_smoothing)
            if model_resume_file is not None:
                # If full path then load from it else from ckpt dir
                self.logger.info(f'Loading model from {f"{model_resume_file}"}')
                model.load_weights(f'{model_resume_file}')

        self.logger.info(model.summary())

        # Some times the model starts overfitting even before one Epoch
        # This is just a hack to reduce # of steps from what they actually are for an Epoch
        # so that best model can be saved in between
        if self.config['dev_mode']:
            n_steps = (self.train_size // self.config['global_batch_size'])
        else:
            n_steps = (self.train_size // self.config['global_batch_size']) // epoch_split_factor

        print(f'# Steps Train: {n_steps}')
        cp_callback = ModelCheckpoint(
            monitor='val_auc',
            mode='max',
            filepath=f'{model_save_file}',
            save_weights_only=True,
            save_best_only=True,
            verbose=1)

        es_cbk = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=patience)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.2,
                                      patience=1, min_lr=1e-8)

        train_history = model.fit(
            train_dataset,
            steps_per_epoch=n_steps,
            validation_data=valid_dataset,
            epochs=self.max_epochs,
            callbacks=[cp_callback, es_cbk, reduce_lr]
        )
        return train_history

    def get_test_data_loader(self, test_file):
        test_data = pd.read_pickle(test_file)
        if self.config['dev_mode']:
            # test_data = test_data.head(10).reset_index(drop=True)
            test_data = test_data[test_data['id'].isin(['0', '1', '10', '100', '1000', '10000'])].reset_index(drop=True)
        self.logger.info(f'# Rows Train: {test_data.shape}')
        test_dataset = (
            tf.data.Dataset
                .from_tensor_slices(
                np.array(test_data['input_ids'].values.tolist()))
                .batch(self.config['global_batch_size'])
        )
        return test_data, test_dataset

    def score_val(self, test_file, model_ckpt, out_file):
        test_data, test_dataset = self.get_test_data_loader(test_file)

        with self.strategy.scope():
            model = build_model(self.config['model_name'], max_len=self.config['max_len'])
            self.logger.info(f'Loading model from {f"{model_ckpt}"}')
            model.load_weights(f'{model_ckpt}')

        test_data['pred_toxic'] = model.predict(test_dataset, verbose=1)
        test_data.to_pickle(out_file)
        return test_data

    def score_test(self, test_file, model_ckpt, out_file):
        self.config['global_batch_size'] = self.config['global_batch_size'] * 4

        test_data, test_dataset = self.get_test_data_loader(test_file)

        with self.strategy.scope():
            model = build_model(self.config['model_name'], max_len=self.config['max_len'])
            self.logger.info(f'Loading model from {f"{model_ckpt}"}')
            model.load_weights(f'{model_ckpt}')

        test_data['pred_toxic'] = model.predict(test_dataset, verbose=1)
        test_data = test_data.rename(columns={'pred_toxic': 'toxic'})
        test_data[['id', 'toxic']].to_csv(out_file, index=False)

    def score_test_split(self, test_file, model_ckpt, out_file):
        self.config['global_batch_size'] = self.config['global_batch_size'] * 4

        test_data, test_dataset = self.get_test_data_loader(test_file)

        with self.strategy.scope():
            model = build_model(self.config['model_name'], max_len=self.config['max_len'])
            self.logger.info(f'Loading model from {f"{model_ckpt}"}')
            model.load_weights(f'{model_ckpt}')

        test_data['pred_toxic'] = model.predict(test_dataset, verbose=1)
        test_data = test_data.rename(columns={'pred_toxic': 'toxic'})
        self.logger.info(f'Pre Rollup Count: {test_data.shape}')
        test_data = test_data.groupby('id')['toxic'].max().reset_index()
        self.logger.info(f'Post Rollup Count: {test_data.shape}')
        test_data[['id', 'toxic']].to_csv(out_file, index=False)

    def score_test_extra(self, test_file, model_ckpt, out_file):
        self.config['global_batch_size'] = self.config['global_batch_size'] * 4

        test_data, test_dataset = self.get_test_data_loader(test_file)

        with self.strategy.scope():
            model = build_model(self.config['model_name'], max_len=self.config['max_len'])
            self.logger.info(f'Loading model from {f"{model_ckpt}"}')
            model.load_weights(f'{model_ckpt}')

        test_data['pred_toxic'] = model.predict(test_dataset, verbose=1)
        test_data = test_data.rename(columns={'pred_toxic': 'toxic'})
        self.logger.info(f'Pre Rollup Count: {test_data.shape}')
        test_data = test_data.groupby('id')['toxic'].mean().reset_index()
        self.logger.info(f'Post Rollup Count: {test_data.shape}')
        test_data[['id', 'toxic']].to_csv(out_file, index=False)