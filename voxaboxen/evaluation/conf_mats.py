import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from voxaboxen.evaluation.raven_utils import Clip


def get_confusion_matrix(predictions_fp, annotations_fp, args, iou, class_threshold):
    c = Clip(label_set=args.label_set, unknown_label=args.unknown_label)

    c.load_predictions(predictions_fp)
    c.threshold_class_predictions(class_threshold)
    c.load_annotations(annotations_fp, label_mapping = args.label_mapping)

    confusion_matrix = {}

    c.compute_matching(IoU_minimum = iou)
    confusion_matrix, confusion_matrix_labels = c.confusion_matrix()

    return confusion_matrix, confusion_matrix_labels

def summarize_confusion_matrix(confusion_matrix, confusion_matrix_labels):
    """ confusion_matrix (dict) : {fp : fp_cm}, where
    fp_cm  : numpy array
    """

    fps = sorted(confusion_matrix.keys())
    l = len(confusion_matrix_labels)

    overall = np.zeros((l, l))

    for fp in fps:
      overall += confusion_matrix[fp]

    return overall, confusion_matrix_labels

def plot_confusion_matrix(data, label_names, target_dir, name=""):
    fig = plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    sns.heatmap(data, annot=True, fmt='d', cmap = 'magma', cbar = True, ax = ax)
    ax.set_title('Confusion Matrix')
    ax.set_yticks([i + 0.5 for i in range(len(label_names))])
    ax.set_yticklabels(label_names, rotation = 0)
    ax.set_xticks([i + 0.5 for i in range(len(label_names))])
    ax.set_xticklabels(label_names, rotation = -90)
    ax.set_ylabel('Prediction')
    ax.set_xlabel('Annotation')
    plt.title(name)

    plt.savefig(os.path.join(target_dir, f"{name}_confusion_matrix.svg"))
    plt.close()
