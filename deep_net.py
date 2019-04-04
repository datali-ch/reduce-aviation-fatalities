import random
from time import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from sklearn.metrics import accuracy_score
import pickle
from snippets import get_random_parameters


def train_neural_net(train_set, test_set, label, training_time, store_itermediate_results):
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

    PARAM_FILE = "deep_net_stats"
    MODEL_FILE = "deep_net_model_"

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

        learning_rate.append(get_random_parameters(PARAM_RANGE["learning_rate"], True, False))
        lr_decay.append(bool(random.getrandbits(1)) * get_random_parameters(PARAM_RANGE["lr_decay"], True, False))
        deep_layers.append(get_random_parameters(PARAM_RANGE["layers"], False, True))

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

        if store_itermediate_results:
            with open(PARAM_FILE, "wb") as f:
                pickle.dump(4, f)
                pickle.dump(learning_rate, f)
                pickle.dump(lr_decay, f)
                pickle.dump(deep_layers, f)
                pickle.dump(accuracy, f)

            curr_model.model.save(MODEL_FILE + str(i) + '.h5')

        i = i + 1

    parameters = [learning_rate, lr_decay, deep_layers]
    return all_models, accuracy, parameters
