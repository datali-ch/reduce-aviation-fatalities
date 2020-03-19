# -*- coding: utf-8 -*-

from time import time

import lightgbm as lgb
import numpy as np
from lightgbm import Booster
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from snippets import get_random_parameters, count_class_occurences, calculate_weights, calculate_single_metric
from typing import Tuple, List
from tensorflow.keras.utils import to_categorical

from config import (
    ALL_CLASSES,
    BAGGING_FRACTION,
    ALL_CLASSES,
    EARLY_STOPPING_ROUND,
    FEATURE_FRACTION,
    MIN_CHILD_WEIGHT,
    NUM_BOOST_ROUND,
    PARAM_RANGE_LGB,
    RECALL_RELEVANT_CLASSES,
)


def train_lgb_model(
    train_set: DataFrame, test_set: DataFrame, label: str, training_time: float, weight_data: bool = True
) -> Tuple[List[Booster], List[list], List[list]]:

    """ Train Light GBM models with randomly generated hyperparameters: learning rate, max bin and number of leaves.
      Model: multiclass classification.

      Args:
          train_set:                                  training set
          test_set:                                   test set, same features as in training set
          label:                                      feature with data labels
          training_time:                              training time (in hours)
          weight_data(optional):                      True for weighting data, False otherwise. Weighting is recommended for imbalanced data.

      Returns:
          all_models:                                  Light GBM models
          metric(list of 2 (N,) lists):                in sample (metric[0]) and out of sample (metric[1]) recall
          parameters(list of 3 (N,) lists):            hyperparameters: learning rates (parameters[0]),
                                                                       max_bin (parameters[1]),
                                                                       num_leaves (parameters[2]),
      """

    all_models = []
    train_metric = []
    test_metric = []
    learning_rate = []
    max_bin = []
    num_leaves = []

    features = list(train_set)
    features.remove(label)
    num_class = len(np.unique(train_set[label]))

    params = {
        "objective": "multiclass",
        "num_class": num_class,
        "metric": "multi_error",
        "min_child_weight": MIN_CHILD_WEIGHT,
        "bagging_fraction": BAGGING_FRACTION,
        "feature_fraction": FEATURE_FRACTION,
        "num_leaves": [],
        "max_bin": [],
        "learning_rate": [],
        "bagging_seed": 420,
        "verbosity": -1,
    }

    if weight_data:
        class2count = count_class_occurences(train_set[label], ALL_CLASSES)
        class_weights = calculate_weights(class2count, ALL_CLASSES)
        metric_weights = calculate_weights(class2count, RECALL_RELEVANT_CLASSES)

        weights_train = [class_weights[row] for row in train_set[label]]
        weights_test = [class_weights[row] for row in test_set[label]]
    else:
        weights_train = None
        weights_test = None

    lgb_train = lgb.Dataset(train_set[features], train_set[label], weight=weights_train)
    lgb_test = lgb.Dataset(test_set[features], test_set[label], weight=weights_test)

    y_true_train = to_categorical(train_set[label])
    y_true_test = to_categorical(test_set[label])

    model_index = 0
    timeout = time() + 60 * 60 * training_time

    while time() < timeout:

        learning_rate.append(
            get_random_parameters(param_range=PARAM_RANGE_LGB["learning_rate"], log_scale=True, is_integer=False)
        )
        max_bin.append(get_random_parameters(param_range=PARAM_RANGE_LGB["max_bin"], log_scale=False, is_integer=True))
        num_leaves.append(
            get_random_parameters(param_range=PARAM_RANGE_LGB["num_leaves"], log_scale=False, is_integer=True)
        )

        params["learning_rate"] = learning_rate[model_index]
        params["max_bin"] = max_bin[model_index]
        params["num_leaves"] = num_leaves[model_index]

        curr_model = lgb.train(
            params,
            lgb_train,
            valid_sets=lgb_test,
            num_boost_round=NUM_BOOST_ROUND,
            early_stopping_rounds=EARLY_STOPPING_ROUND,
            verbose_eval=False,
        )
        all_models.append(curr_model)

        recall_train = calculate_single_metric(
            train_set[features], y_true_train, curr_model, metric_weights, iteration=curr_model.best_iteration
        )
        train_metric.append(recall_train)

        recall_test = calculate_single_metric(
            test_set[features], y_true_test, curr_model, metric_weights, iteration=curr_model.best_iteration
        )
        test_metric.append(recall_test)

    metric = [train_metric, test_metric]
    parameters = [learning_rate, max_bin, num_leaves]
    return all_models, metric, parameters
