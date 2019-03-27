training_time = 0.05 # in hours
allFeatures = True

if allFeatures:
    features = features_n 
else:
    features = features_n[:22]


if 'learning_rate' not in locals():
    i = 0
    accuracy_train_lgb = []
    accuracy_test_lgb = []
    
    learning_rate = []
    max_bin = []
    num_leaves = []
 
timeout = time.time() + 60*60*training_time


while time.time() < timeout:
    
    learning_rate.append(10**(-6*np.random.rand()))
    max_bin.append(int(10**np.random.uniform(1, 3)))
    num_leaves.append(int(np.random.uniform(10, 10*len(features))))
    
    params = {"objective" : "multiclass",
                  "num_class": 4,
                  "metric" : "multi_error", #'multi_logloss'
                  "num_leaves" : num_leaves[i],
                  "max_bin": max_bin[i],
                  "learning_rate" : learning_rate[i],
                  "min_child_weight" : 50,
                  "bagging_fraction" : 0.7,
                  "feature_fraction" : 0.7,
                  "bagging_seed" : 420,
                  "verbosity" : -1
                 }
        
    lg_train = lgb.Dataset(train_set[features], label=train_set["event"])
    lg_test = lgb.Dataset(test_set[features], label=test_set["event"])
    lgb_model = lgb.train(params, lg_train, 10000, valid_sets=[lg_test], early_stopping_rounds=50, verbose_eval=100)
      
    pred_train = lgb_model.predict(train_set[features_n], num_iteration=lgb_model.best_iteration)
    pred_test = lgb_model.predict(test_set[features_n], num_iteration=lgb_model.best_iteration)
    
    predicted_train = np.argmax(pred_train, axis=1)
    predicted_test = np.argmax(pred_test, axis=1)
    
    accuracy_train_lgb = np.append(accuracy_train_lgb, accuracy_score(labels_train, predicted_train))
    accuracy_test_lgb = np.append(accuracy_test_lgb, accuracy_score(labels_test, predicted_test))
    
    i = i + 1