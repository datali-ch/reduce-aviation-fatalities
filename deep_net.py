import random
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
from time import time
from sklearn.metrics import accuracy_score


def train_neural_net(train_set, test_set, label, training_time):
    features = list(train_set)
    features.remove(label)

    curr_epochs = 10
    curr_batch = 1000

    alpha = []
    deep_layers = []
    lr_decay = []
    BATCH_NORM = []

    accuracy_train = []
    accuracy_test = []

    i = 0
    timeout = time() + 60*60*training_time
    while time() < timeout:

        alpha.append(10 ** (-6 * np.random.rand()))
        deep_layers.append(int(10 ** (np.random.uniform(1, 5))))
        lr_decay.append(bool(random.getrandbits(1)) * (10 ** (np.random.uniform(-8, 0))))
        BATCH_NORM.append(bool(random.getrandbits(1)))

        # instantiate model
        model = tf.keras.models.Sequential()

        # Input layers
        model.add(layers.Dense(10, input_dim=train_set.shape[1] - 1))
        if BATCH_NORM[i]:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        # Intermediary layers
        model.add(layers.Dense(deep_layers[i]))
        if BATCH_NORM[i]:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        # Output layer
        model.add(layers.Dense(len(np.unique(train_set[label])) + 1))
        if BATCH_NORM[i]:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation('softmax'))

        adamOptimizer = tf.keras.optimizers.Adam(lr=alpha[i], beta_1=0.9, beta_2=0.999, epsilon=None,
                                                 decay=lr_decay[i],
                                                 amsgrad=False);
        model.compile(optimizer=adamOptimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(train_set[features], tf.keras.utils.to_categorical(train_set[label]),
                            epochs=curr_epochs,
                            batch_size=curr_batch)

        predicted_train = np.argmax(model.predict(train_set[features]), axis=1)
        predicted_test = np.argmax(model.predict(test_set[features]), axis=1)

        accuracy_train = np.append(accuracy_train, accuracy_score(train_set[label], predicted_train))
        accuracy_test = np.append(accuracy_test, accuracy_score(test_set[label], predicted_test))

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

    return history.model
