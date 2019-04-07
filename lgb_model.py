# -*- coding: utf-8 -*-

from time import time

import lightgbm as lgb
import numpy as np
from lightgbm import Booster
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from snippets import get_random_parameters
from typing import Tuple, List

from config import (BAGGING_FRACTION, EARLY_STOPPING_ROUND, FEATURE_FRACTION,
                     MIN_CHILD_WEIGHT, NUM_BOOST_ROUND, NUM_CLASS,
                     PARAM_RANGE_LGB)


def train_lgb_model(train_set: DataFrame, test_set: DataFrame, label: str, training_time: float) -> \
    Tuple[List[Booster], List[list], List[list]]:

    """ Train Light GBM models with randomly generated hyperparameters: learning rate, max bin and number of leaves.
      Model: multiclass classification.

      Args:
          train_set(pandas df):                       training set
          test_set(pandas df):                        test set, same features as in training set
          label(str):                                 feature with data labels
          training_time(float):                       training time (in hours)

      Returns:
          all_models(list of lightgbm.basic.Booster):  Light GBM models
          accuracy(list of 2 (N,) lists):              in sample (accuracy[0]) and out of sample (accuracy[1]) accuracy
          parameters(list of 3 (N,) lists):            hyperparameters: learning rates (parameters[0]),
                                                                       max_bin (parameters[1]),
                                                                       num_leaves (parameters[2]),
      """

    all_models = []
    train_acc = []
    test_acc = []
    learning_rate = []
    max_bin = []
    num_leaves = []

    features = list(train_set)
    features.remove(label)

    params = {
        "objective": "multiclass",
        "num_class": NUM_CLASS,
        "metric": "multi_error",
        "min_child_weight": MIN_CHILD_WEIGHT,
        "bagging_fraction": BAGGING_FRACTION,
        "feature_fraction": FEATURE_FRACTION,
        "num_leaves": [],
        "max_bin": [],
        "learning_rate": [],
        "bagging_seed": 420,
        "verbosity": -1
    }

    model_index = 0
    timeout = time() + 60*60*training_time

    while time() < timeout:

        learning_rate.append(get_random_parameters(
            param_range=PARAM_RANGE_LGB["learning_rate"], 
            log_scale=True, 
            is_integer=False))
        max_bin.append(get_random_parameters(
            param_range=PARAM_RANGE_LGB["max_bin"], 
            log_scale=False, 
            is_integer=True))
        num_leaves.append(get_random_parameters(
            param_range=PARAM_RANGE_LGB["num_leaves"],
            log_scale=False, 
            is_integer=True))

        params["learning_rate"] = learning_rate[model_index]
        params["max_bin"] = max_bin[model_index]
        params["num_leaves"] = num_leaves[model_index]

        lgb_train = lgb.Dataset(train_set[features], train_set[label])
        lgb_test = lgb.Dataset(test_set[features], test_set[label])

        curr_model = lgb.train(params, lgb_train, valid_sets=lgb_test,
                               num_boost_round=NUM_BOOST_ROUND,
                               early_stopping_rounds=EARLY_STOPPING_ROUND,
                               verbose_eval=False)
        all_models.append(curr_model)


        predicted = np.argmax(curr_model.predict(train_set[features], num_iteration=curr_model.best_iteration), axis = 1)
        train_acc.append(accuracy_score(train_set[label], predicted))

        predicted = np.argmax(curr_model.predict(test_set[features], num_iteration=curr_model.best_iteration), axis = 1)
        test_acc.append(accuracy_score(test_set[label], predicted))

    accuracy = [train_acc, test_acc]
    parameters = [learning_rate, max_bin, num_leaves]
    return all_models, accuracy, parameters