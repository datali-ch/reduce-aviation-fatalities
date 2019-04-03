import pandas as pd
from sklearn.model_selection import train_test_split
from snippets import *

def loadData(file, sample_size, process_signals):

    TEST_SIZE = 0.2
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

    df = pd.read_csv(file, nrows=5*sample_size, dtype=DTYPES)
    df = df.sample(n=5*sample_size)

    # Process physiological data
    df['pilot'] = 100 * df['seat'] + df['crew']

    if process_signals:
        add_respiration_rate(df)
        add_heart_rate(df)

        MONTAGES = ['longitudial_bipolar', 'cz_reference', 'crossed_bipolar']
        for montage in MONTAGES:
            process_eeg_data(df, montage)

    df = df.dropna()

    # Normalize features
    irrelevant_fields = ['crew', 'seat', 'time', 'experiment', 'event', 'pilot']
    features_n = [item for item in list(df) if item not in irrelevant_fields]

    if df.shape[0] < sample_size:
        raise Exception('Your sample is too small to process signals')
    df = df.sample(n=sample_size)
    data = normalize_by_pilots(df, features_n)

    # Label outcome states
    dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    data["event"] = data["event"].apply(lambda x: dic[x])

    # Delete irrelevant fields
    irrelevant_fields = ['crew', 'seat', 'time', 'experiment']
    data = data.drop(irrelevant_fields, axis=1)

    # Get test and train set
    train_set, test_set = train_test_split(data, test_size=TEST_SIZE, random_state=666)

    return train_set, test_set

    """
    # Initiate list with results
    estimation_ts = []
    """
