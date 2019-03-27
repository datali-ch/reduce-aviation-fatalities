# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:44:42 2019

@author: surowka
"""

# Plot normalized confusion matrix - in sample
plot_confusion_matrix(labels_train, predicted_train, classes=[0,1,2,3], normalize=True,
                      title='Normalized confusion matrix')

# Plot normalized confusion matrix - out of sample
plot_confusion_matrix(labels_test, predicted_test, classes=[0,1,2,3], normalize=True,
                      title='Normalized confusion matrix')

plt.show()

"""
# Show feature importance
fig, ax = plt.subplots(figsize=(12,10))
lgb.plot_importance(lgb_model, height=0.8, ax=ax)
ax.grid(False)
plt.ylabel('Feature', size=12)
plt.xlabel('Importance', size=12)
plt.title("Importance of the Features of our LightGBM Model", fontsize=15)
plt.show()
"""
