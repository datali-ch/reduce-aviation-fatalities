# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from snippets import *


def load_data(file, sample_size=None, process_signals=False):
    """ Load train.csv dataset from https://www.kaggle.com/c/reducing-commercial-aviation-fatalities

    Args:
        file(str):                          data file stored locally, full path
        sample_size(int, optional):         number of randomly sampled observations from file. If None,
                                            all observations will be included.
        process_signals(bool, optional):    True to process physiological data, False otherwise.
    Returns:
        train_set(pandas df):               df with N out of M observations from data file
        test_set(pandas df):                df with M-N observations from data file
    """

    TEST_SIZE = 0.2
    LABEL = "event"
    TRAINING_IRRELEVANT = ['crew', 'seat', 'time', 'experiment']
    LABEL_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    DTYPES = {"crew": "int8",
              "experiment": "category",
              "time": "float32",
              "seat": "int8",
              "eeg_fp1": "float32",
              "eeg_f7": "float32",
              "eeg_f8": "float32",
              "eeg_t4": "float32",
              "eeg_t6": "float32",
              "eeg_t5": "float32",
              "eeg_t3": "float32",
              "eeg_fp2": "float32",
              "eeg_o1": "float32",
              "eeg_p3": "float32",
              "eeg_pz": "float32",
              "eeg_f3": "float32",
              "eeg_fz": "float32",
              "eeg_f4": "float32",
              "eeg_c4": "float32",
              "eeg_p4": "float32",
              "eeg_poz": "float32",
              "eeg_c3": "float32",
              "eeg_cz": "float32",
              "eeg_o2": "float32",
              "ecg": "float32",
              "r": "float32",
              "gsr": "float32",
              "event": "category",
              }

    df = pd.read_csv(file, dtype=DTYPES)

    # Unique identifier for pilot
    df['pilot'] = 100 * df['seat'] + df['crew']
    TRAINING_IRRELEVANT.append('pilot')

    # Process physiological data
    if process_signals:
        add_respiration_rate(df)
        add_heart_rate(df)

        MONTAGES = ['longitudial_bipolar', 'cz_reference', 'crossed_bipolar']
        for montage in MONTAGES:
            process_eeg_data(df, montage)

    df = df.dropna()
    if df.shape[0] < sample_size:
        raise Exception('Your sample is too small to process signals')

    # Prepare data as trainig set
    features_n = [item for item in list(df) if item not in TRAINING_IRRELEVANT + [LABEL]]
    grouping_feature = "pilot"
    if sample_size is not None:
        df = df.sample(n=min(sample_size, df.shape[0]))
    data = normalize_data(df, grouping_feature, features_n)
    data = data.drop(TRAINING_IRRELEVANT, axis=1)
    data[LABEL] = data[LABEL].apply(lambda x: LABEL_MAP[x])

    train_set, test_set = train_test_split(data, test_size=TEST_SIZE, random_state=666)

    return train_set, test_set
