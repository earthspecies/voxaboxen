"""
Function to set up project-level metadata
"""

import argparse
import os
import sys
from typing import List, Union

import pandas as pd

from voxaboxen.project.params import parse_project_args, save_params


def project_setup(args: Union[argparse.Namespace, List[str]]) -> None:
    args = parse_project_args(args)

    if not os.path.exists(args.project_dir):
        os.makedirs(args.project_dir)

    all_annots = []
    for info_fp in [args.train_info_fp, args.val_info_fp, args.test_info_fp]:
        if info_fp is None:
            continue

        info = pd.read_csv(info_fp)
        annot_fps = list(info["selection_table_fp"])

        for annot_fp in annot_fps:
            if annot_fp != "None":
                selection_table = pd.read_csv(annot_fp, delimiter="\t")
                annots = list(selection_table["Annotation"].astype(str))
                all_annots.extend(annots)

    label_set = sorted(set(all_annots))
    label_mapping = {x: x for x in label_set}
    label_mapping["Unknown"] = "Unknown"
    unknown_label = "Unknown"

    if unknown_label in label_set:
        label_set.remove(unknown_label)

    args.label_set = label_set
    args.label_mapping = label_mapping
    args.unknown_label = unknown_label

    save_params(args)


if __name__ == "__main__":
    project_setup(sys.argv[1:])
