"""
Functions for creating confusion matrices
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from voxaboxen.evaluation.raven_utils import Clip


def get_confusion_matrix(
    predictions_fp: str,
    annotations_fp: str,
    args: argparse.Namespace,
    iou: float,
    class_threshold: float,
) -> Tuple[np.ndarray, List[str]]:
    """
    Produces a confusion matrix for predictions on one audio file
    Parameters
    ----------
    predictions_fp : str
        path to predictions selection table
    annotations_fp : str
        path to annotations selection table
    args : argparse.Namespace
        Configuration arguments containing model parameters
    iou : float
        IoU value for matching predictions with annotations
    class_threshold : float
        Value under which class predictions will be counted as Unknown
    Returns
    -------
    confusion_matrix : numpy.ndarray
    confusion_matrix_labels : list
        list of str for confusion matrix labels
    """
    c = Clip(label_set=args.label_set, unknown_label=args.unknown_label)

    c.load_predictions(predictions_fp)
    c.threshold_class_predictions(class_threshold)
    c.load_annotations(annotations_fp, label_mapping=args.label_mapping)

    confusion_matrix = {}

    c.compute_matching(IoU_minimum=iou)
    confusion_matrix, confusion_matrix_labels = c.confusion_matrix()

    return confusion_matrix, confusion_matrix_labels


def summarize_confusion_matrix(
    confusion_matrix: Dict[str, np.ndarray], confusion_matrix_labels: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregates multiple per-file confusion matrices
    Parameters
    ----------
    confusion_matrix : dict
        dict of the form {fp : fp_cm}, where
        fp_cm is numpy array giving the confusion matrix
    confusion_matrix_labels : list
    Returns
    -------
    overall : numpy.ndarray
    confusion_matrix_labels : list
    """

    fps = sorted(confusion_matrix.keys())
    n_labels = len(confusion_matrix_labels)

    overall = np.zeros((n_labels, n_labels))

    for fp in fps:
        overall += confusion_matrix[fp]

    return overall, confusion_matrix_labels


def plot_confusion_matrix(
    data: np.ndarray, label_names: List[str], target_dir: str, name: str = ""
) -> None:
    """
    Plots confusion matrix
    """
    fig = plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor="w", edgecolor="k")
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    sns.heatmap(data, annot=True, fmt="d", cmap="magma", cbar=True, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_yticks([i + 0.5 for i in range(len(label_names))])
    ax.set_yticklabels(label_names, rotation=0)
    ax.set_xticks([i + 0.5 for i in range(len(label_names))])
    ax.set_xticklabels(label_names, rotation=-90)
    ax.set_ylabel("Prediction")
    ax.set_xlabel("Annotation")
    plt.title(name)

    plt.savefig(os.path.join(target_dir, f"{name}_confusion_matrix.svg"))
    plt.close()
