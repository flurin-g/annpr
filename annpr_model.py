from time import time
from os import path

import tensorflow as tf
import keras as ks
import numpy as np

from pairwise_kl_divergence import pairwise_kl_divergence
from data_generator import DataGenerator
from vox_utils import get_all_sets_50m_50w
from definitions import TRAIN_CONF, WEIGHTS_PATH, LOG_DIR

INPUT_DIMS = [TRAIN_CONF['input_data']['mel_spectrogram_x'],
              TRAIN_CONF['input_data']['mel_spectrogram_y']]


def create_lstm(units: int, gpu: bool, name: str, is_sequence: bool = True):
    if gpu:
        return ks.layers.CuDNNLSTM(units, return_sequences=is_sequence, input_shape=INPUT_DIMS, name=name)
    else:
        return ks.layers.LSTM(units, return_sequences=is_sequence, input_shape=INPUT_DIMS, unroll=True, name=name)


def build_optimizer():
    optimizer = None
    p = TRAIN_CONF['topology']['optimizer']
    if p['type'] == 'adam':
        optimizer = ks.optimizers.Adam(p['learning_rate'],
                                       p['beta_1'],
                                       p['beta_2'],
                                       float(p['epsilon']),
                                       p['decay'])

    if p['type'] == 'rms':
        optimizer = ks.optimizers.RMSprop()

    return optimizer


def build_model(mode: str = "train"):
    topology = TRAIN_CONF['topology']
    is_gpu = tf.test.is_gpu_available(cuda_only=True)

    model = ks.Sequential()
    model.add(
        ks.layers.Bidirectional(create_lstm(topology['blstm1_units'], is_gpu, name='blstm_1'), input_shape=INPUT_DIMS))

    model.add(ks.layers.Dropout(topology['dropout1']))

    model.add(ks.layers.Bidirectional(create_lstm(topology['blstm2_units'], is_gpu, is_sequence=False, name='blstm_2')))
    if mode == 'extraction':
        return model

    num_units = topology['dense1_units']
    model.add(ks.layers.Dense(num_units, name='dense_1'))

    model.add(ks.layers.Dropout(topology['dropout2']))

    num_units = topology['dense2_units']
    model.add(ks.layers.Dense(num_units, name='dense_2'))

    num_units = topology['dense3_units']
    model.add(ks.layers.Dense(num_units, name='dense_3'))

    model.add(ks.layers.Softmax(num_units, name='softmax'))

    adam = build_optimizer()
    model.compile(loss=pairwise_kl_divergence,
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


def train_model(create_spectrograms: bool = False, weights_path: str = WEIGHTS_PATH):
    model_dir = path.dirname(WEIGHTS_PATH)
    checkpoint_pattern = path.join(model_dir, 'weights.{epoch:02d}-{val_loss:.2f}-' + str(time()) + '.hdf5')

    callbacks = [
        ks.callbacks.ProgbarLogger('steps'),
        ks.callbacks.ModelCheckpoint(checkpoint_pattern),
        ks.callbacks.TensorBoard(
            LOG_DIR,
            histogram_freq=1,
            write_grads=True,
            write_images=False,
            write_graph=True
        )
    ]

    input_data = TRAIN_CONF['input_data']
    batch_size = input_data['batch_size']

    train_set, dev_set = get_all_sets_50m_50w(create_spectrograms)

    training_generator = DataGenerator(train_set, INPUT_DIMS, batch_size)

    val_data = DataGenerator.generate_batch(dev_set, batch_size, INPUT_DIMS[0], INPUT_DIMS[1])

    annpr = build_model()
    annpr.summary()
    annpr.fit_generator(generator=training_generator,
                        epochs=input_data['epochs'],
                        validation_data=val_data,
                        use_multiprocessing=True,
                        callbacks=callbacks)

    annpr.save_weights(weights_path, overwrite=True)
