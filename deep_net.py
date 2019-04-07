# -*- coding: utf-8 -*-

import random
from time import time

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from config import (BATCH_NORM, BATCH_SIZE, BETA_1, BETA_2, EPOCHS,
                    INPUT_LAYERS, MODEL_FILE, PARAM_FILE,
                    PARAM_RANGE_PERCEPTRON)
from keras import Model
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from snippets import get_random_parameters
from typing import List, Tuple
import json


def train_neural_net(train_set: DataFrame, test_set: DataFrame, label: str, training_time: float, 
                     store_intermediate_results: bool = False) -> Tuple[List[Model], List[list], List[list]]:
    """ Train neural networks with randomly generated hyperparameters: learning rate, learning rate
        decay and number of layer. Model: fully connected NN for multiclass classification.

        Args:
            train_set(pandas df):                       training set
            test_set(pandas df):                        test set, same features as in training set
            label(str):                                 feature with data labels
            training_time(float):                       training time (in hours)
            store_intermediate_results(bool, optional): True for saving model at the end of training, False otherwise

        Returns:
            all_models(list of keras.model):            neural networks models
            accuracy(list of 2 (N,) lists):             in sample (accuracy[0]) and out of sample (accuracy[1]) accuracy
            parameters(list of 3 (N,) lists):           hyperparameters: learning rates (parameters[0]),
                                                                         learning rate decays (parameters[1]),
                                                                         num fo layers (parameters[2]),
    """

    all_models = []
    train_acc = []
    test_acc = []
    learning_rate = []
    lr_decay = []
    deep_layers = []

    features = list(train_set)
    features.remove(label)
    num_class = len(np.unique(train_set[label]))

    model_index = 0
    timeout = time() + 60 * 60 * training_time

    while time() < timeout:

        learning_rate.append(get_random_parameters(
            param_range=PARAM_RANGE_PERCEPTRON["learning_rate"],
            log_scale=True,
            is_integer=False))
        lr_decay.append(random.getrandbits(1) * get_random_parameters(
            param_range=PARAM_RANGE_PERCEPTRON["lr_decay"],
            log_scale=True,
            is_integer=False))
        deep_layers.append(get_random_parameters(
            param_range=PARAM_RANGE_PERCEPTRON["layers"],
            log_scale=False,
            is_integer=True))

        # instantiate model
        model = tf.keras.models.Sequential()

        # Input layers
        model.add(layers.Dense(INPUT_LAYERS, input_dim=train_set[features].shape[1]))
        if BATCH_NORM:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        # Intermediary layers
        model.add(layers.Dense(deep_layers[model_index]))
        if BATCH_NORM:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        # Output layer
        model.add(layers.Dense(num_class))
        if BATCH_NORM:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation('softmax'))

        adam_optimizer = tf.keras.optimizers.Adam(lr=learning_rate[model_index],
                                                  decay=lr_decay[model_index],
                                                  beta_1=BETA_1,
                                                  beta_2=BETA_2,
                                                  epsilon=None,
                                                  amsgrad=False)
        model.compile(optimizer=adam_optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        curr_model = model.fit(train_set[features], tf.keras.utils.to_categorical(train_set[label]), epochs=EPOCHS,
                               batch_size=BATCH_SIZE, verbose=0)
        all_models.append(curr_model)

        predicted = np.argmax(model.predict(train_set[features]), axis=1)
        train_acc.append(accuracy_score(train_set[label], predicted))

        predicted = np.argmax(model.predict(test_set[features]), axis=1)
        test_acc.append(accuracy_score(test_set[label], predicted))

        if store_intermediate_results:
            intermediate_results = {
                "learning_rate": learning_rate,
                "lr_decay": lr_decay,
                "deep_layers": deep_layers,
                "accuracy": [train_acc, test_acc]
            }

            with open(PARAM_FILE, "w") as f:
                json.dump(intermediate_results, f, indent=4)

            curr_model.model.save(MODEL_FILE + str(model_index) + '.h5')

        model_index = model_index + 1

    accuracy = [train_acc, test_acc]
    parameters = [learning_rate, lr_decay, deep_layers]
    return all_models, accuracy, parameters
