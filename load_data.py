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

    df = pd.read_csv(file, dtype=DTYPES)
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

    # Delete irrelevant fields
    # Label outcome states
    dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    data["event"] = data["event"].apply(lambda x: dic[x])

    irrelevant_fields = ['crew', 'seat', 'time', 'experiment']
    data = data.drop(irrelevant_fields, axis=1)

    # Get test and train set
    train_set, test_set = train_test_split(data, test_size=TEST_SIZE, random_state=666)

    # Process data shape
    labels_train = train_set.loc[:, "event"].get_values().copy()
    labels_test = test_set.loc[:, "event"].get_values().copy()

    features_train = train_set.drop("event", axis=1).values.copy()
    features_test = test_set.drop("event", axis=1).values.copy()

    return features_train, features_test, labels_train, labels_test

    """
    # Initiate list with results
    estimation_ts = []
    accuracy_train = np.empty(0)
    accuracy_test = np.empty(0)
    """

    """
    ## Get only original values
    original_features = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2", "ecg", "r", "gsr", "event"]
    original_dataset = data[original_features]

    train_set_orig, test_set_orig = train_test_split(original_dataset, test_size=0.2, random_state=666)

    ## Process data shape
    labels_train = train_set_orig.loc[:,"event"].get_values().copy()
    labels_test = test_set_orig.loc[:,"event"].get_values().copy()

    features_train = train_set_orig.drop("event", axis = 1).values.copy()
    features_test = test_set_orig.drop("event", axis = 1).values.copy()

    """