# -*- coding: utf-8 -*-

import glob
import json
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from biosppy.signals import ecg, resp
from keras import Model
from lightgbm import Booster
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from typing import Dict, List, Tuple, Union


def normalize_data(df: DataFrame, grouping_feature: str, features_to_scale: Union[list, str]) -> DataFrame:
    """ Normalize dataset separately for each value of grouping feature

    Args:
        df(pandas df):                 dataset to scale
        grouping_feature(str):         feature by which df will be split to perform normalization
        features_to_scale(list of str):features to normalize

    Returns:
        normalized_df(pandas df):      df with normalized features_to_scale
    """

    normalized_df = df.copy()
    pilots = normalized_df[grouping_feature].unique()

    for pilot in pilots:
        ids = normalized_df[normalized_df[grouping_feature] == pilot].index
        scaler = MinMaxScaler()
        normalized_df.loc[ids, features_to_scale] = scaler.fit_transform(normalized_df.loc[ids, features_to_scale])

    return normalized_df


def get_random_parameters(param_range: Tuple[float, float]=None, log_scale: bool=True, is_integer: bool=False, **kwargs) -> Union[float, int]:
    """ Simulate random numbers in given range
    Args:
        param_range(tuple of len=2):        range of parameter
        log_scale(bool):                    scale of parameter. True for log scale, False for uniform scale
        is_integer(bool):                   format of parameter. True for integer, False for real number

    Returns:
        param(float or int):                randomly simulated parameter within param_range on log/uniform scale
    """

    if log_scale:
        param = 10 ** (np.random.uniform(*np.log10(param_range)))
    else:
        param = np.random.uniform(*param_range)

    if is_integer:
        param = int(param)

    return param


def process_eeg_data(df: DataFrame, type: Union[list, str]) -> None:
    """ Process EEG signal to get potential difference. Use one of clinically used montages. Add potential
    differences to input.

    Args:
        df(pandas df):      dataset with ECG signal labelled as 'ecg'
        type(list of str):  EEG montages. Supported values: "longitudial_bipolar", "cz_reference", "crossed_bipolar"

    """

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


def add_respiration_rate(df: DataFrame) -> None:
    """ Process chest movement signal to calculate respiration rate. Add the respiration rate to input as
    'respiration_rate' feature.

    Args:
        df(pandas df):                 dataset with chest movement signal labelled as 'r'
    """

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


def import_perceptron_stats(file_name: str) -> Dict[str, list]:
    """ Import hyperparameters and accuracy for perceptron models estimated and saved with train_neural_net()

    Args:
        file_name(str):                     model data file generated with train_neural_net(), full path

    Returns:
        dict:                               hyperparameters and accuracy for stored perceptron models. Consists of:
            learning_rate((N,) list):       learning rates
            lr_decay((N,) list):            learning rate decaya
            deep_layers((N,) list):         number of fully connected layers in NN
            accuracy(list of 2 (N,) lists): in sample (accuracy[0]) and out of sample (accuracy[1]) accuracy
    """

    with open(file_name, "rb") as f:
        return json.load(f)

def import_perceptron_models(directory: str) -> List[Model]:
    """ Import all keras models saved in given directory

    Args:
        directory(str):                     folder directory where keras models are stored

    Returns:
        all_models(list of keras.model):    neural networks models
    """

    os.chdir(directory)
    all_models = []
    for file in glob.glob("*.h5"):
        curr_model = load_model(file)
        all_models.append(curr_model)

    return all_models


def plot_feature_importance(lgb_model: Booster, show: bool=True) -> Axes:
    """ Plot feature importance of Light GBM Model

    Args:
        lgb_model(lightgbm Booster):          trained Light GBM Model
        show(bool, optional):                 display image. True for showing the image, False otherwise

    Returns:
        ax(matplotlib axes):                  fig axes
    """

    fig, ax = plt.subplots(figsize=(12, 10))
    lgb.plot_importance(lgb_model, height=0.8, ax=ax)
    ax.grid(False)
    plt.ylabel('Feature', size=12)
    plt.xlabel('Importance', size=12)
    plt.title("Feature Importance in LightGBM Model", fontsize=15)
    if show:
        plt.draw()
        plt.pause(0.05)
    return ax


def plot_training_progress(perceptron_models: List[Model], indices: ndarray, metric: str, show: bool=True) -> Axes:
    """ Plot log loss or accuracy of neural network during training

    Args:
        perceptron_models(list of keras.model):    neural networks models
        indices((N,) np array):                    indices of models to display
        metric(str):                               model quality metric. 'loss' for log loss, 'accuracy' for accuracy
        show(bool, optional):                      display image. True for showing the image, False otherwise

    Returns:
        ax(matplotlib axes.Axes):                  fig axes
    """

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
        plt.draw()
        plt.pause(0.05)

    return ax


def add_heart_rate(df: DataFrame) -> None:
    """ Process ECG signal to calculate heart rate. Add the heart rate to input as 'heart_rate' feature.

    Args:
        df(pandas df):                 dataset with ECG signal labelled as 'ecg'
    """

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


def plot_confusion_matrix(y_true: ndarray, y_pred: ndarray, classes: list,
                          show: bool=True,
                          normalize: bool=False,
                          title: str=None,
                          cmap: Colormap=plt.cm.Blues) -> Axes:
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
        plt.draw()
        plt.pause(0.05)
    return ax
