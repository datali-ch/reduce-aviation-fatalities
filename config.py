# -*- coding: utf-8 -*-

# General parameters
LABEL = "event"

ALL_CLASSES = [0, 1, 2, 3]
RECALL_RELEVANT_CLASSES = [1, 3]
DATA2CLASS = {"A": 0, "B": 1, "C": 2, "D": 3}  # A = Baseline, B = Surprise, C = Focus, D = Distraction
CLASS2LABEL = {0: "Baseline", 1: "Surprise", 2: "Focus", 3: "Distraction"}

DATA_FILE = "train.csv"
SAMPLE_SIZE = 100000
PROCESS_SIGNALS = False

DEFAULT_TRAINING_TIME_PERCEPTRON = 0.33
DEFAULT_TRAINING_TIME_LGB = 0.5

# Config parameters for load_data()
TEST_SIZE = 0.2
TRAINING_IRRELEVANT = ["crew", "seat", "time", "experiment"]

DTYPES = {
    "crew": "int8",
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

# Config parameters for train_neural_net()
PARAM_RANGE_PERCEPTRON = {"learning_rate": (1e-1, 1e-6), "layers": (10, 100), "lr_decay": (0.5, 1e-8)}
BATCH_NORM = False
EPOCHS = 1000
BATCH_SIZE = 80000
BETA_1 = 0.9
BETA_2 = 0.999
INPUT_LAYERS = 10

PARAM_FILE = "deep_net_stats.json"
MODEL_FILE = "deep_net_model_"

SAVE_INTERMEDIATE_RESULTS = True
MODELS_TO_PLOT = 5

# Config parameters for train_lgb_model()
PARAM_RANGE_LGB = {"learning_rate": (1e-1, 1e-6), "max_bin": (10, 1e3), "num_leaves": (10, 250)}
MIN_CHILD_WEIGHT = 50
BAGGING_FRACTION = 0.7
FEATURE_FRACTION = 0.7
EARLY_STOPPING_ROUND = 50
NUM_BOOST_ROUND = 10000
