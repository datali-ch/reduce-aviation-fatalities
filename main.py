from load_data import *
from lgb_model import *
from deep_net import *

file = 'C:/Users/surowka/Documents/reduce-aviation-fatalities/train.csv'
sample_size = 1000
process_signals = False
train_set, test_set = load_data(file, sample_size, process_signals)

training_time = 0.01
label = "event"

#deep_networks, accuracy = train_neural_net(train_set, test_set, label, training_time, True)
#lgb_models, accuracy = train_lgb_model(train_set, test_set, label, training_time)
