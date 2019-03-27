# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:04:39 2019

@author: surowka

with open("modelStats", "wb") as f:
    pickle.dump(3, f)
    pickle.dump(accuracy_train, f)
    pickle.dump(accuracy_test, f)
    pickle.dump(estimation_ts, f)
"""
import pickle 
from tensorflow.keras.models import load_model

PRINT_LEARNING_CURVES = False
    
modelStats = []
with open("modelStats", "rb") as f:
    for _ in range(pickle.load(f)):
        modelStats.append(pickle.load(f))

accuracy_train = modelStats[0]
accuracy_test = modelStats[1]
estimation_ts = modelStats[2]

#del modelStats

allModels = []
for i in range(len(accuracy_test)):
    model = load_model('models/model_' + str(i) + '.h5')
    allModels.append(model)
    
## PRINT DECAY RATES
learning_rate = []
for i in range(len(allModels)):
    curr_alpha = allModels[i].optimizer.get_config()["lr"]
    learning_rate.append(curr_alpha)
    
    if PRINT_LEARNING_CURVES:
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        fig.set_size_inches(30, 15, forward=True)
        loss = estimation_ts[i][:,0]
        ax.plot(range(1,len(loss)+1), loss)
        plt.ylabel('Log loss')
        plt.xlabel('Epochs')
        plt.title('Learning rate:' + str(round(curr_alpha,6)))
        filename = 'plots/log_loss' + str(i) +'.png'
        fig.savefig(filename)  # save the figure to file
        plt.close(fig) 
