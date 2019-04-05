from load_data import *
from lgb_model import *
from deep_net import *
from snippets import *

data_file = 'C:/Users/surowka/Documents/reduce-aviation-fatalities/train.csv'
sample_size = 100000
process_signals = False
train_set, test_set = load_data(data_file, sample_size, process_signals)

label = "event"
features = list(train_set)
features.remove(label)
y_true = test_set[label]

# Train models
training_time = 0.1   # Training time in hours
deep_networks, accuracy_deep_net, _ = train_neural_net(train_set, test_set, label, training_time)
lgb_models, accuracy_lgb, _ = train_lgb_model(train_set, test_set, label, training_time)

# Results - Fully Connected NN
ind = np.argsort(-accuracy_deep_net[1])
best_neural_net = deep_networks[ind[0]]
y_pred = np.argmax(best_neural_net.model.predict(test_set[features]), axis=1)

models_to_plot = 3
plot_training_progress(deep_networks, ind[:models_to_plot], 'loss')
plot_confusion_matrix(y_true, y_pred, classes=[0,1,2,3], normalize=True,
                      title='Best Neural Network \n Normalized confusion matrix')


# Results - Light GBM
ind = np.argsort(-accuracy_lgb[1])
best_lbg = lgb_models[ind[0]]
y_pred = np.argmax(best_lbg.predict(test_set[features], num_iteration=best_lbg.best_iteration), axis=1)

plot_confusion_matrix(y_true, y_pred, classes=[0,1,2,3], normalize=True,
                      title='Best LightGBM Model\n Normalized confusion matrix')
plot_feature_importance(best_lbg, True)

