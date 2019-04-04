import random
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
from time import time
from sklearn.metrics import accuracy_score


def train_neural_net(train_set, test_set, label, training_time):
    PARAM_RANGE = {
        "learning_rate": (1e-1, 1e-6),
        "layers": (10, 1e3),
        "lr_decay": (0.5, 1e-8)
    }
    BATCH_NORM = False
    EPOCHS = 10
    BATCH_SIZE = 1000
    BETA_1 = 0.9
    BETA_2 = 0.999

    all_models = []
    accuracy = [[], []]
    learning_rate = []
    lr_decay = []
    deep_layers = []

    features = list(train_set)
    features.remove(label)
    num_class = len(np.unique(train_set[label])) + 1

    i = 0
    timeout = time() + 60 * 60 * training_time

    while time() < timeout:

        learning_rate.append(get_random_parameters(PARAM_RANGE["learning_rate"], True))
        lr_decay.append(bool(random.getrandbits(1)) * get_random_parameters(PARAM_RANGE["lr_decay"], True))
        deep_layers.append(get_random_parameters(PARAM_RANGE["layers"], False))

        # instantiate model
        model = tf.keras.models.Sequential()

        # Input layers
        model.add(layers.Dense(10, input_dim=train_set.shape[1] - 1))
        if BATCH_NORM:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        # Intermediary layers
        model.add(layers.Dense(deep_layers[i]))
        if BATCH_NORM:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        # Output layer
        model.add(layers.Dense(num_class))
        if BATCH_NORM:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation('softmax'))

        adamOptimizer = tf.keras.optimizers.Adam(lr=learning_rate[i],
                                                 decay=lr_decay[i],
                                                 beta_1=BETA_1,
                                                 beta_2=BETA_2,
                                                 epsilon=None,
                                                 amsgrad=False)
        model.compile(optimizer=adamOptimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        curr_model = model.fit(train_set[features], tf.keras.utils.to_categorical(train_set[label]), epochs=EPOCHS,
                               batch_size=BATCH_SIZE, verbose=0)
        all_models.append(curr_model)

        j = 0
        for dataset in (train_set, test_set):
            predicted = np.argmax(model.predict(dataset[features]), axis=1)
            accuracy[j] = np.append(accuracy[j], accuracy_score(dataset[label], predicted))
            j = j + 1

        """
        curr_stats = np.column_stack((history.history["loss"], history.history["acc"]))
        estimation_ts.append(curr_stats)
        
        with open("modelStats", "wb") as f:
            pickle.dump(7, f)
            pickle.dump(accuracy_train, f)
            pickle.dump(accuracy_test, f)
            pickle.dump(estimation_ts, f)
            pickle.dump(alpha, f)
            pickle.dump(deep_layers, f)
            pickle.dump(lr_decay, f)
            pickle.dump(BATCH_NORM, f)
        
        history.model.save('models/model_' + str(i) + '.h5')
        """
        i = i + 1

    parameters = [learning_rate, lr_decay, deep_layers]
    return all_models, accuracy, parameters


def get_random_parameters(param_range, log_scale):
    if log_scale:
        param = 10 ** (np.random.uniform(*np.log10(param_range)))
    else:
        param = int(np.random.uniform(*param_range))

    return param
