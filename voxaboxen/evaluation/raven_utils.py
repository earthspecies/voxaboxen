"""
Code for working with audio and Raven selection table annotations together
"""

import warnings
from typing import Dict, List, Optional, Set, Tuple, Union

import librosa
import numpy as np
import pandas as pd

import voxaboxen.evaluation.metrics as metrics


class Clip:
    """
    A class representing an audio clip and its associated
    metadata, annotations, and predictions.

    Parameters
    ----------
    label_set : set or list, optional
        A collection of known labels that annotations and predictions can belong to.
    unknown_label : str or any, optional
        A label used to denote unknown or unlabeled data.

    Attributes
    ----------
    sr : int or None
        Sampling rate of the audio clip in Hz.
    samples : np.ndarray or None
        Audio samples as a NumPy array.
    duration : float or None
        Duration of the audio clip in seconds.
    annotations : pandas.DataFrame or None
        Ground-truth annotations for the audio clip.
    predictions : pandas.DataFrame or None
        Predicted annotations for the audio clip.
    matching : list or None
        Information about how predictions and annotations are matched.
    matched_annotations : list or None
        Subset of annotations that have been matched to predictions.
    matched_predictions : list or None
        Subset of predictions that have been matched to annotations.
    label_set : set or list or None
        The set of valid labels used for annotations and predictions.
    unknown_label : str or None
        The label used to represent unknown or missing categories.
    """

    def __init__(
        self,
        label_set: Optional[Union[Set[str], List[str]]] = None,
        unknown_label: Optional[str] = None,
    ) -> None:
        self.sr = None
        self.samples = None
        self.duration = None
        self.annotations = None
        self.predictions = None
        self.matching = None
        self.matched_annotations = None
        self.matched_predictions = None
        self.label_set = label_set
        self.unknown_label = unknown_label

    def load_selection_table(
        self,
        fp: str,
        view: Optional[str] = None,
        label_mapping: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Load a Raven selection table from a file.

        Parameters
        ----------
        fp : str
            Filepath to the selection table.
        view : str, optional
            View to filter by (e.g., "Waveform" or "Spectrogram").
            If not provided and multiple views are present, a warning is issued.
        label_mapping : dict, optional
            Dictionary mapping old labels to new labels. If provided, annotations not
            in the mapping keys are dropped.

        Returns
        -------
        pd.DataFrame
            Filtered and/or mapped selection table annotations.
        """

        annotations = pd.read_csv(fp, delimiter="\t")
        if view is None and "View" in annotations:
            views = annotations["View"].unique()
            if len(views) > 1:
                warnings.warn(
                    (
                        "I found more than one view in selection table. To avoid double"
                        f" counting, pass view as a parameter. Views found: {view}"
                    ),
                    stacklevel=2,
                )

        if view is not None:
            annotations = annotations[
                annotations["View"].str.contains("Waveform")
            ].reset_index()

        if label_mapping is not None:
            annotations["Annotation"] = annotations["Annotation"].map(label_mapping)
            annotations = annotations[~pd.isnull(annotations["Annotation"])]

        return annotations

    def load_audio(self, fp: str) -> None:
        """
        Load audio from file
        """
        self.samples, self.sr = librosa.load(fp, sr=None)
        self.duration = len(self.samples) / self.sr

    def load_annotations(
        self,
        fp: str,
        view: Optional[str] = None,
        label_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        load annotation selection table from file
        """
        self.annotations = self.load_selection_table(
            fp, view=view, label_mapping=label_mapping
        )
        self.annotations["index"] = self.annotations.index

    def threshold_class_predictions(self, class_threshold: float) -> None:
        """
        If class probability is below a threshold, switch label to unknown
        """

        assert self.unknown_label is not None
        self.predictions.loc[
            self.predictions["Class Prob"] < class_threshold, "Annotation"
        ] = self.unknown_label

    def load_predictions(
        self,
        fp: str,
        view: Optional[str] = None,
        label_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        load prediction selection table from file
        """
        self.predictions = self.load_selection_table(
            fp, view=view, label_mapping=label_mapping
        )
        self.predictions["index"] = self.predictions.index

    def compute_matching(self, IoU_minimum: float = 0.5) -> None:
        """Bipartite graph matching between predictions and annotations
        Maximizes the number of matchings with IoU > IoU_minimum
        Saves a list of indexes of matched pairs
        """
        ref = np.array(self.annotations[["Begin Time (s)", "End Time (s)"]]).T
        est = np.array(self.predictions[["Begin Time (s)", "End Time (s)"]]).T
        self.matching = metrics.match_events(
            ref, est, min_iou=IoU_minimum, method="fast"
        )
        self.matched_annotations = [p[0] for p in self.matching]
        self.matched_predictions = [p[1] for p in self.matching]

    def evaluate(self) -> Dict:
        """
        Count true positive, false positive, and false negative predictions

        Returns
        ------
        Dict containing counts of tp, fp, and fn's for the audio file
        """
        eval_sr = 50
        dur_samples = int(self.duration * eval_sr)  # compute frame-wise metrics at 50Hz

        if self.label_set is None:
            TP = len(self.matching)
            FP = len(self.predictions) - TP
            FN = len(self.annotations) - TP

            # segmentation-based metrics
            seg_annotations = np.zeros((dur_samples,))
            seg_predictions = np.zeros((dur_samples,))

            for _i, row in self.annotations.iterrows():
                start_sample = int(row["Begin Time (s)"] * eval_sr)
                end_sample = min(int(row["End Time (s)"] * eval_sr), dur_samples)
                seg_annotations[start_sample:end_sample] = 1

            for _i, row in self.predictions.iterrows():
                start_sample = int(row["Begin Time (s)"] * eval_sr)
                end_sample = min(int(row["End Time (s)"] * eval_sr), dur_samples)
                seg_predictions[start_sample:end_sample] = 1

            TP_seg = int((seg_predictions * seg_annotations).sum())
            FP_seg = int((seg_predictions * (1 - seg_annotations)).sum())
            FN_seg = int(((1 - seg_predictions) * seg_annotations).sum())

            return {
                "all": {
                    "TP": TP,
                    "FP": FP,
                    "FN": FN,
                    "TP_seg": TP_seg,
                    "FP_seg": FP_seg,
                    "FN_seg": FN_seg,
                }
            }

        else:
            out = {
                label: {
                    "TP": 0,
                    "FP": 0,
                    "FN": 0,
                    "TP_seg": 0,
                    "FP_seg": 0,
                    "FN_seg": 0,
                }
                for label in self.label_set
            }
            pred_label = np.array(self.predictions["Annotation"])
            annot_label = np.array(self.annotations["Annotation"])
            for p in self.matching:
                annotation = annot_label[p[0]]
                prediction = pred_label[p[1]]

                if self.unknown_label is not None and prediction == self.unknown_label:
                    pass  # treat predicted unknowns as no predictions for these metrics
                elif annotation == prediction:
                    out[annotation]["TP"] += 1
                elif (
                    self.unknown_label is not None and annotation == self.unknown_label
                ):
                    out[prediction]["FP"] -= 1  # adjust FP for unknown labels

            for label in self.label_set:
                n_annot = int((annot_label == label).sum())
                n_pred = int((pred_label == label).sum())
                out[label]["FP"] = out[label]["FP"] + n_pred - out[label]["TP"]
                out[label]["FN"] = out[label]["FN"] + n_annot - out[label]["TP"]

                # segmentation-based metrics
                seg_annotations = np.zeros((dur_samples,))
                seg_predictions = np.zeros((dur_samples,))

                annot_sub = self.annotations[self.annotations["Annotation"] == label]
                pred_sub = self.predictions[self.predictions["Annotation"] == label]

                begins = (annot_sub["Begin Time (s)"] * eval_sr).astype(int)
                ends = (annot_sub["End Time (s)"] * eval_sr).astype(int)
                for b, e in zip(begins, ends, strict=False):
                    seg_annotations[b:e] = 1

                begins = (pred_sub["Begin Time (s)"] * eval_sr).astype(int)
                ends = (pred_sub["End Time (s)"] * eval_sr).astype(int)
                for b, e in zip(begins, ends, strict=False):
                    seg_predictions[b:e] = 1

                TP_seg = int((seg_predictions * seg_annotations).sum())
                FP_seg = int((seg_predictions * (1 - seg_annotations)).sum())
                FN_seg = int(((1 - seg_predictions) * seg_annotations).sum())
                out[label]["TP_seg"] = TP_seg
                out[label]["FP_seg"] = FP_seg
                out[label]["FN_seg"] = FN_seg

            return out

    def confusion_matrix(self) -> Optional[Tuple[np.ndarray, List[str]]]:
        """
        Compute confusion matrix for predictions vs. annotations.

        Returns
        -------
        Optional[Tuple[np.ndarray, List[str]]]
            A tuple containing:
                - confusion_matrix: np.ndarray of shape (num_classes+1, num_classes+1)
                - confusion_matrix_labels: list of label names
                including 'None' and optionally the unknown label
            Returns None if self.label_set is None.
        """
        if self.label_set is None:
            return None
        else:
            confusion_matrix_labels = self.label_set.copy()
            if self.unknown_label is not None:
                confusion_matrix_labels.append(self.unknown_label)
            confusion_matrix_labels.append("None")
            confusion_matrix_size = len(confusion_matrix_labels)

            confusion_matrix = np.zeros((confusion_matrix_size, confusion_matrix_size))
            cm_nobox_idx = confusion_matrix_labels.index("None")

            pred_label = np.array(self.predictions["Annotation"])
            annot_label = np.array(self.annotations["Annotation"])

            for p in self.matching:
                annotation = annot_label[p[0]]
                prediction = pred_label[p[1]]
                cm_annot_idx = confusion_matrix_labels.index(annotation)
                cm_pred_idx = confusion_matrix_labels.index(prediction)
                confusion_matrix[cm_pred_idx, cm_annot_idx] += 1

            for label in confusion_matrix_labels:
                if label == "None":
                    continue

                # count false positive and false negative dets, regardless of class
                cm_label_idx = confusion_matrix_labels.index(label)

                # fp
                n_pred = int((pred_label == label).sum())
                n_positive_detections_row = confusion_matrix.sum(1)[cm_label_idx]
                n_false_detections = n_pred - n_positive_detections_row
                confusion_matrix[cm_label_idx, cm_nobox_idx] = n_false_detections

                # fn
                n_annot = int((annot_label == label).sum())
                n_positive_detections_col = confusion_matrix.sum(0)[cm_label_idx]
                n_missed_detections = n_annot - n_positive_detections_col
                confusion_matrix[cm_nobox_idx, cm_label_idx] = n_missed_detections

        return confusion_matrix, confusion_matrix_labels
