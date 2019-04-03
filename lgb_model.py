import lightgbm as lgb
from sklearn.metrics import accuracy_score
import numpy as np
from time import time


def train_lgb_model(train_set, test_set, label, training_time):
    accuracy_train = []
    accuracy_test = []

    learning_rate = []
    max_bin = []
    num_leaves = []

    i = 0
    timeout = time() + 60*60*training_time

    while time() < timeout:
        learning_rate.append(10 ** (-6 * np.random.rand()))
        max_bin.append(int(10 ** np.random.uniform(1, 3)))
        num_leaves.append(int(np.random.uniform(10, 10 * train_set.shape[1] - 1)))

        features = list(train_set)
        features.remove(label)

        params = {"objective": "multiclass",
                  "num_class": 4,
                  "metric": "multi_error",
                  "num_leaves": num_leaves[i],
                  "max_bin": max_bin[i],
                  "learning_rate": learning_rate[i],
                  "min_child_weight": 50,
                  "bagging_fraction": 0.7,
                  "feature_fraction": 0.7,
                  "bagging_seed": 420,
                  "verbosity": -1
                  }

        lgb_train = lgb.Dataset(train_set[features], train_set[label])
        lgb_test = lgb.Dataset(test_set[features], test_set[label])

        lgb_model = lgb.train(params, lgb_train, 10000, valid_sets=lgb_test, early_stopping_rounds=50, verbose_eval=100)

        pred_train = lgb_model.predict(train_set[features], num_iteration=lgb_model.best_iteration)
        pred_test = lgb_model.predict(test_set[features], num_iteration=lgb_model.best_iteration)

        predicted_train = np.argmax(pred_train, axis=1)
        predicted_test = np.argmax(pred_test, axis=1)

        accuracy_train = np.append(accuracy_train, accuracy_score(train_set[label], predicted_train))
        accuracy_test = np.append(accuracy_test, accuracy_score(test_set[label], predicted_test))

    return lgb_model
