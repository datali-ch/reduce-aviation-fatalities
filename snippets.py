# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:48:24 2019

@author: surowka
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import glob, os
from biosppy.signals import ecg, resp
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import lightgbm as lgb
import pickle


def normalize_by_pilots(df, features_to_scale):
    curr_df = df.copy()
    pilots = curr_df["pilot"].unique()

    for pilot in pilots:
        ids = curr_df[curr_df["pilot"] == pilot].index
        scaler = MinMaxScaler()
        curr_df.loc[ids, features_to_scale] = scaler.fit_transform(curr_df.loc[ids, features_to_scale])

    return curr_df


def get_random_parameters(param_range, log_scale, is_integer):
    if log_scale:
        param = 10 ** (np.random.uniform(*np.log10(param_range)))
    else:
        param = np.random.uniform(*param_range)

    if is_integer:
        param = int(param)

    return param


def process_eeg_data(df, type):
    if type == 'longitudial_bipolar':

        electrodes_from = ['eeg_fp1', 'eeg_f7', 'eeg_t3', 'eeg_t5', 'eeg_fp1', 'eeg_f3', 'eeg_c3', 'eeg_p3', 'eeg_fz',
                           'eeg_cz', 'eeg_fp2', 'eeg_f8', 'eeg_t4', 'eeg_t6', 'eeg_fp2', 'eeg_f4', 'eeg_c4', 'eeg_p4']
        electrodes_to = ['eeg_f7', 'eeg_t3', 'eeg_t5', 'eeg_o1', 'eeg_f3', 'eeg_c3', 'eeg_p3', 'eeg_o1', 'eeg_cz',
                         'eeg_pz', 'eeg_f8', 'eeg_t4', 'eeg_t6', 'eeg_o2', 'eeg_f4', 'eeg_c4', 'eeg_p4', 'eeg_o2']

    elif type == 'cz_reference':

        electrodes_from = ['eeg_fp1', 'eeg_fp2', 'eeg_f8', 'eeg_t4', 'eeg_c4', 'eeg_t6', 'eeg_p4', 'eeg_o2', 'eeg_o1',
                           'eeg_p3', 'eeg_t5', 'eeg_c3', 'eeg_t3', 'eeg_f7', 'eeg_f3', 'eeg_fp1']
        electrodes_to = ['eeg_cz'] * len(electrodes_from)

    elif type == 'crossed_bipolar':

        electrodes_from = ['eeg_f7', 'eeg_f3', 'eeg_fz', 'eeg_t3', 'eeg_c3', 'eeg_cz', 'eeg_p3', 'eeg_t5', 'eeg_o1']
        electrodes_to = ['eeg_f8', 'eeg_fz', 'eeg_f4', 'eeg_t4', 'eeg_cz', 'eeg_c4', 'eeg_p4', 'eeg_t6', 'eeg_o2']

    for i in range(len(electrodes_from)):
        name_1 = electrodes_from[i]
        name_2 = electrodes_to[i]

        col_name = name_1[4:] + '_' + name_2[4:]
        df[col_name] = df[name_1] - df[name_2]
        df[col_name].astype('float32')


def add_respiration_rate(df):
    df["respiration_rate"] = np.nan

    all_pilots = df.pilot.unique()
    all_experiments = df.experiment.unique()

    for pilot in all_pilots:
        for experiment in all_experiments:

            where_in_df = df.index[(df.pilot == pilot) & (df.experiment == experiment)]
            subset = df.loc[where_in_df, ['time', 'r']]

            try:
                subset.sort_values(by='time')

                out = resp.resp(signal=subset['r'], sampling_rate=256, show=False)
                where_in_subset = out['resp_rate_ts']
                resp_rate = out['resp_rate']

                global_ind = where_in_df[where_in_subset]
                for i in range(len(resp_rate)):
                    df.loc[global_ind[i]:global_ind[i + 1], ['respiration_rate']] = resp_rate[i]
            except:
                print('Not all respiration rates were calculated')

    df["respiration_rate"].astype('float32')


def import_perceptron_stats(file_name):
    model_stats = []
    with open(file_name, "rb") as f:
        for _ in range(pickle.load(f)):
            model_stats.append(pickle.load(f))

    dict = {"learning_rate": model_stats[0],
            "lr_decay": model_stats[1],
            "deep_layers": model_stats[2],
            "accuracy": model_stats[3]}

    return dict


def import_perceptron_models(directory):
    os.chdir(directory)
    all_models = []
    for file in glob.glob("*.h5"):
        curr_model = load_model(file)
        all_models.append(curr_model)

    return all_models


def plot_feature_importance(lgb_model, show=True):
    fig, ax = plt.subplots(figsize=(12, 10))
    lgb.plot_importance(lgb_model, height=0.8, ax=ax)
    ax.grid(False)
    plt.ylabel('Feature', size=12)
    plt.xlabel('Importance', size=12)
    plt.title("Feature Importance in LightGBM Model", fontsize=15)
    if show:
        plt.show()
    return ax


def plot_training_progress(perceptron_models, indices, metric, show=True):

    fig, ax = plt.subplots(figsize=(12, 10))
    for ind in indices:
        ts = perceptron_models[ind].history[metric]
        ax.plot(range(1, len(ts) + 1), ts, label='Model ' + str(ind))

    plt.xlabel('Epochs')
    if metric is 'loss':
        plt.ylabel('Log loss')
        plt.title('Model loss during training', fontsize=15)
        plt.legend(loc='upper right')
    else:
        plt.ylabel('Accuracy')
        plt.title('Model accuracy during training', fontsize=15)
        plt.legend(loc='lower right')

    if show:
        plt.show()

    return ax


def add_heart_rate(df):
    df["heart_rate"] = np.nan

    all_pilots = df.pilot.unique()
    all_experiments = df.experiment.unique()

    for pilot in all_pilots:
        for experiment in all_experiments:

            where_in_df = df.index[(df.pilot == pilot) & (df.experiment == experiment)]
            subset = df.loc[where_in_df, ['time', 'ecg']]

            try:
                subset.sort_values(by='time')

                out = ecg.ecg(signal=subset['ecg'], sampling_rate=256, show=False)
                where_in_subset = out['heart_rate_ts']
                heart_rate = out['heart_rate']

                global_ind = where_in_df[where_in_subset]
                for i in range(len(heart_rate)):
                    df.loc[global_ind[i]:global_ind[i + 1], ['heart_rate']] = heart_rate[i]

            except:
                print('Not all respiration rates were calculated')

    df["heart_rate"].astype('float32')


def plot_confusion_matrix(y_true, y_pred, classes,
                          show=True,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    Code from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if show:
        plt.show()
    return ax