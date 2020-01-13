# -*- coding: utf-8 -*-
"""
This script addresses the problem of detecting distraction state of a pilot.
Using physiological data, it attempts to detect whether a pilot is distracted, sleepy or
in other dangerous cognitive state. The full problem definition is described at:
https://www.kaggle.com/c/reducing-commercial-aviation-fatalities/overview

Data needed can be downloaded under the same address.

IMPORTANT: Training machine learning models requires substantial computational resources.
           Make sure to adjust sample_size and training_time to your needs

Author: Magdalena Surowka
        Data Scientist | Machine Learning Specialist
        magdalena.surowka@gmail.com
"""

import argparse
import numpy as np
from config import (
    ALL_CLASSES,
    CLASS2LABEL,
    DATA_FILE,
    LABEL,
    PROCESS_SIGNALS,
    SAMPLE_SIZE,
    SAVE_INTERMEDIATE_RESULTS,
    DEFAULT_TRAINING_TIME_LGB,
    DEFAULT_TRAINING_TIME_PERCEPTRON,
    MODELS_TO_PLOT,
    RECALL_RELEVANT_CLASSES,
)
from deep_net import train_neural_net
from lgb_model import train_lgb_model
from load_data import load_data
from matplotlib import pyplot as plt
from snippets import plot_confusion_matrix, plot_feature_importance, plot_training_progress


def main(args: argparse.Namespace) -> None:
    """Main function
    """
    train_set, test_set = load_data(DATA_FILE, SAMPLE_SIZE, PROCESS_SIGNALS)

    features = list(train_set)
    features.remove(LABEL)
    y_true = test_set[LABEL]

    # Train models
    deep_networks, recall_deep_net, _ = train_neural_net(
        train_set, test_set, LABEL, args.training_time_perceptron, store_intermediate_results=args.save
    )
    lgb_models, recall_lgb, _ = train_lgb_model(train_set, test_set, LABEL, args.training_time_lgb)

    # Results - Fully Connected NN
    ind = np.argsort(-1 * np.array(recall_deep_net[1]))
    best_neural_net = deep_networks[ind[0]]
    y_pred = np.argmax(best_neural_net.model.predict(test_set[features]), axis=1)

    models_to_plot = min([MODELS_TO_PLOT, len(deep_networks)])
    plot_training_progress(deep_networks, ind[:models_to_plot], "loss")
    plot_training_progress(deep_networks, ind[:models_to_plot], "weighted_acc")

    plot_confusion_matrix(
        y_true, y_pred, CLASS2LABEL, normalize=True, title="Normalized confusion matrix"
    )

    # Results - Light GBM

    ind = np.argsort(-1 * np.array(recall_lgb[1]))
    best_lbg = lgb_models[ind[0]]
    y_pred = np.argmax(best_lbg.predict(test_set[features], num_iteration=best_lbg.best_iteration), axis=1)
    plot_confusion_matrix(
        y_true, y_pred, CLASS2LABEL, normalize=True, title="Best LightGBM Model\n Normalized confusion matrix"
    )
    plot_feature_importance(best_lbg, True)



def get_arg_parser():
    parser = argparse.ArgumentParser(
        description="This script addresses the problem of detecting distraction state of a pilot."
    )

    parser.add_argument("--save", action="store_true", help="Save the intermediaite results")
    parser.add_argument(
        "--training-time-perceptron",
        type=float,
        default=DEFAULT_TRAINING_TIME_PERCEPTRON,
        help="Training time for fully connected neural network (in hours)",
    )
    parser.add_argument(
        "--training-time-lgb",
        type=float,
        default=DEFAULT_TRAINING_TIME_LGB,
        help="Training time for Light GBM model (in hours)",
    )

    return parser


if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    main(args)
    plt.show()  # Keep plots open after main will have finished
