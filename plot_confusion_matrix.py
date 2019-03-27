# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:44:42 2019

@author: surowka


# Plot normalized confusion matrix - in sample
plot_confusion_matrix(labels_train, predicted_train, classes=[0,1,2,3], normalize=True,
                      title='Normalized confusion matrix')

# Plot normalized confusion matrix - out of sample
plot_confusion_matrix(labels_test, predicted_test, classes=[0,1,2,3], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# Show feature importance
fig, ax = plt.subplots(figsize=(12,10))
lgb.plot_importance(lgb_model, height=0.8, ax=ax)
ax.grid(False)
plt.ylabel('Feature', size=12)
plt.xlabel('Importance', size=12)
plt.title("Importance of the Features of our LightGBM Model", fontsize=15)
plt.show()
"""

import matplotlib.pyplot as plt
 
# Data to plot
fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
#fig.set_size_inches(30, 15, forward=True)


labels = 'Startle/Surprise', 'Base', 'Channelized Attention', 'Diverted Attention', 
sizes = [2686, 58171, 34150, 4993]
#colors = ['gold', 'blue', 'skyblue', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode 1st slice
 
# Plot
ax.pie(sizes, explode=explode, labels=labels,
autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()

fig.savefig('plots/labels_repartition.png')  # save the figure to file
plt.close(fig) 
