import lightgbm as lgb
from sklearn.metrics import accuracy_score
import numpy as np
from time import time


def train_lgb_model(train_set, test_set, label, training_time):

    alpha = {
        "range": (1e-1, 1e-6),
        "log_scale": True
    }

    beta = {
        "range": (10, 1e3),
        "log_scale": True
    }

    gamma = {
        "range": (10, 10*train_set.shape[1]-1),
        "log_scale": False
    }

    accuracy = [[], []]
    all_models = []

    learning_rate = []
    max_bin = []
    num_leaves = []

    i = 0
    timeout = time() + 60*60*training_time

    while time() < timeout:

        if alpha["log_scale"]:
            learning_rate.append(10 ** (np.random.uniform(*np.log10(alpha["range"]))))
        else:
            learning_rate.append(np.random.uniform(*alpha["range"]))

        if beta["log_scale"]:
            max_bin.append(int(10 ** (np.random.uniform(*np.log10(beta["range"])))))
        else:
            max_bin.append(int(np.random.uniform(*beta["range"])))

        if gamma["log_scale"]:
            num_leaves.append(int(10 ** (np.random.uniform(*np.log10(gamma["range"])))))
        else:
            num_leaves.append(int(np.random.uniform(*gamma["range"])))

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

        curr_model = lgb.train(params, lgb_train, 10000, valid_sets=lgb_test, early_stopping_rounds=50,
                               verbose_eval=100)
        all_models.append(curr_model)

        j = 0
        for dataset in (train_set, test_set):

            probabilities = curr_model.predict(dataset[features], num_iteration=curr_model.best_iteration)
            predicted_labels = np.argmax(probabilities, axis=1)
            accuracy[i] = np.append(accuracy[i], accuracy_score(dataset[label], predicted_labels))
            i = j+1

    parameters = [learning_rate, max_bin, num_leaves]
    return all_models, accuracy, parameters