# -*- coding: utf-8 -*-

import random
from time import time

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from config import (
    BATCH_NORM,
    BETA_1,
    BETA_2,
    EPOCHS,
    BATCH_SIZE,
    ALL_CLASSES,
    INPUT_LAYERS,
    MODEL_FILE,
    PARAM_FILE,
    PARAM_RANGE_PERCEPTRON,
    RECALL_RELEVANT_CLASSES,
)
from keras import Model
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
from snippets import get_random_parameters
from typing import List, Tuple
import json
from keras import backend as K
from snippets import calculate_single_metric, count_class_occurences, calculate_weights


def train_neural_net(
    train_set: DataFrame,
    test_set: DataFrame,
    label: str,
    training_time: float,
    weight_data: bool = True,
    store_intermediate_results: bool = False,
) -> Tuple[List[Model], List[list], List[list]]:
    """ Train neural networks with randomly generated hyperparameters: learning rate, learning rate
        decay and number of layer. Model: fully connected NN for multiclass classification.

        Args:
            train_set:                              training set
            test_set:                               test set, same features as in training set
            label                                   feature with data labels
            training_time                           training time (in hours)
            store_intermediate_results(optional):   True for saving model at the end of training, False otherwise
            weight_data(optional):                  True for weighting data, False otherwise. Weighting is recommended for imbalanced data.


        Returns:
            all_models:                             neural networks models
            metric(list of 2 (N,) lists):           in sample (metric[0]) and out of sample (metric[1]) recall
            parameters(list of 3 (N,) lists):       hyperparameters: learning rates (parameters[0]),
                                                                     learning rate decays (parameters[1]),
                                                                     num fo layers (parameters[2]),
    """

    all_models = []
    train_metric = []
    test_metric = []
    learning_rate = []
    lr_decay = []
    deep_layers = []

    features = list(train_set)
    features.remove(label)
    num_class = len(np.unique(train_set[label]))

    if weight_data:
        class2count = count_class_occurences(train_set[label], ALL_CLASSES)
        class_weights = calculate_weights(class2count, ALL_CLASSES)
        metric_weights = calculate_weights(class2count, RECALL_RELEVANT_CLASSES)
    else:
        class_weights = None
        metric_weights = dict(zip(range(num_class), [1] * num_class))

    y_true_train = tf.keras.utils.to_categorical(train_set[label])
    y_true_test = tf.keras.utils.to_categorical(test_set[label])

    model_index = 0
    timeout = time() + 60 * 60 * training_time

    while time() < timeout:

        learning_rate.append(
            get_random_parameters(param_range=PARAM_RANGE_PERCEPTRON["learning_rate"], log_scale=True, is_integer=False)
        )
        lr_decay.append(
            random.getrandbits(1)
            * get_random_parameters(param_range=PARAM_RANGE_PERCEPTRON["lr_decay"], log_scale=True, is_integer=False)
        )
        deep_layers.append(
            get_random_parameters(param_range=PARAM_RANGE_PERCEPTRON["layers"], log_scale=False, is_integer=True)
        )

        # instantiate model
        model = tf.keras.models.Sequential()

        # Input layers
        model.add(layers.Dense(INPUT_LAYERS, input_dim=train_set[features].shape[1]))
        if BATCH_NORM:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))

        # Intermediary layers
        model.add(layers.Dense(deep_layers[model_index]))
        if BATCH_NORM:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))

        # Output layer
        model.add(layers.Dense(num_class))
        if BATCH_NORM:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation("softmax"))

        adam_optimizer = tf.keras.optimizers.Adam(
            lr=learning_rate[model_index],
            decay=lr_decay[model_index],
            beta_1=BETA_1,
            beta_2=BETA_2,
            epsilon=None,
            amsgrad=False,
        )
        model.compile(optimizer=adam_optimizer, loss="categorical_crossentropy", weighted_metrics=["accuracy"])

        curr_model = model.fit(
            train_set[features],
            y_true_train,
            class_weight=class_weights,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )
        all_models.append(curr_model)

        recall_train = calculate_single_metric(train_set[features], y_true_train, model, metric_weights)
        train_metric.append(recall_train)

        recall_test = calculate_single_metric(test_set[features], y_true_test, model, metric_weights)
        test_metric.append(recall_test)

        if store_intermediate_results:
            intermediate_results = {
                "learning_rate": learning_rate,
                "lr_decay": lr_decay,
                "deep_layers": deep_layers,
                "evaluation_metric": [train_metric, test_metric],
            }

            with open(PARAM_FILE, "w") as f:
                json.dump(intermediate_results, f, indent=4)

            curr_model.model.save(MODEL_FILE + str(model_index) + ".h5")

        model_index = model_index + 1

    metric = [train_metric, test_metric]
    parameters = [learning_rate, lr_decay, deep_layers]
    return all_models, metric, parameters
