# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:53:26 2019

@author: surowka


# Import data
if len(accuracy_train) == 0:
    import loadSavedData
    
# Index to save file
import glob
model_files = glob.glob("models/*")
all_indices = [model_files[i][-4] for i in range(len(model_files))]
i = np.asarray(all_indices, dtype=int).max() + 1 
    
"""
import random
import tensorflow.keras.layers as layers

curr_epochs = 100
curr_batch = 1000
i = 0
alpha = []
deep_layers = []
lr_decay = []
BATCH_NORM =[]

timeout = time.time() + 60*0.01*60 

while time.time() < timeout:
    
    alpha.append(10**(-6*np.random.rand()))
    deep_layers.append(int(10**(np.random.uniform(1, 5))))
    lr_decay.append(bool(random.getrandbits(1))*(10**(np.random.uniform(-8, 0))))
    BATCH_NORM.append(bool(random.getrandbits(1)))
    
    parameters = [alpha, layers, lr_decay, BATCH_NORM]
     
    # instantiate model
    model = tf.keras.models.Sequential()
    
    # Input layers
    model.add(layers.Dense(10, input_dim=features_train.shape[1]))
    if BATCH_NORM[i]:
        model.add(layers.BatchNormalization()) 
    model.add(layers.Activation('relu'))
    
    # Intermediary layers
    model.add(layers.Dense(deep_layers[i]))
    #model.add(layers.BatchNormalization()) # May work better when generalizing to out of sample
    model.add(layers.Activation('relu'))
    
    # output layer
    model.add(layers.Dense(4))
    if BATCH_NORM[i]:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation('softmax'))
    
    adamOptimizer = tf.keras.optimizers.Adam(lr=alpha[i], beta_1=0.9, beta_2=0.999, epsilon=None, decay=lr_decay[i], amsgrad=False);
    model.compile(optimizer=adamOptimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(features_train, to_categorical(labels_train), epochs=curr_epochs, batch_size=curr_batch) 
    
    
    predicted_train = np.argmax(model.predict(features_train), axis=1)
    predicted_test = np.argmax(model.predict(features_test), axis=1)
     # In sample accuracy: 76% under 10 epochs
    # 84% under 30 epochs, 85.7% under 50 epochs, 88-89% under 100 epochs, 89.5% under 200 epochs
    
    accuracy_train = np.append(accuracy_train, accuracy_score(labels_train, predicted_train))
    accuracy_test = np.append(accuracy_test, accuracy_score(labels_test, predicted_test))
    
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
    i = i+1