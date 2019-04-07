# -*- coding: utf-8 -*-

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from config import DTYPES, LABEL, LABEL_MAP, TEST_SIZE, TRAINING_IRRELEVANT
from snippets import (add_heart_rate, add_respiration_rate, normalize_data,
                       process_eeg_data)


def load_data(file: str, sample_size: int=None, process_signals: bool=False) -> DataFrame:

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
        raise ValueError('Your sample is too small to process signals')

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
