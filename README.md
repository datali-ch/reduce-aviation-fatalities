# Reduce Aviation Fatalities

Most flight-related fatalities stem from a loss of “airplane state awareness.” That is, ineffective attention management on the part of pilots who may be distracted, sleepy or in other dangerous cognitive states.

The challenge is to build a model to detect troubling events from aircrew’s physiological data. We use the data acquired from actual pilots in test situations, and your models should be able to run calculations in real time to monitor the cognitive states of pilots. With your help, pilots could then be alerted when they enter a troubling state, preventing accidents and saving lives.

Reducing aircraft fatalities is just one of the complex problems that Booz Allen Hamilton has been solving for business, government, and military leaders for over 100 years. Through devotion, candor, courage, and character, they produce original solutions where there are no roadmaps. Now you can help them find answers, save lives, and change the world.

SETUP:
* Download data (train.csv) from https://www.kaggle.com/c/reducing-commercial-aviation-fatalities/data
* Parametrize training_time in main.py file. The quality of your model will depend purely on the training time
* The most relevant hyperparameters will be included in the search
* All other can be defined in config file
* Neural network require much longer training time than LGB model. You can pass those arguments separately
* If you plan to train neural network for a longer time, consider saving your intermediate results with relevant 
parameters to train_neural_net()
