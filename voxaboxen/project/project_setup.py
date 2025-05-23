import os
import sys
import pandas as pd
from voxaboxen.project.params import save_params, parse_project_args

def project_setup(args):
    """
    Set up project files prior to training
    Parameters
    ----------
    args : argparse.Namespace
        Configuration arguments; see params.py
    Returns
    ----------
    """
    args = parse_project_args(args)

    if not os.path.exists(args.project_dir):
        os.makedirs(args.project_dir)

    all_annots = []
    for info_fp in [args.train_info_fp, args.val_info_fp, args.test_info_fp]:
        #if info_fp is None:
        if not os.path.exists(info_fp): # what's the use case for not having one of these info_fps?
            continue

        info = pd.read_csv(info_fp)
        extended_annot_fps = [x if x.startswith(args.data_dir) else os.path.join(args.data_dir, x) for x in info.selection_table_fp]
        assert all(os.path.exists(x) for x in extended_annot_fps)
        info.selection_table_fp = extended_annot_fps

        extended_audio_fps = [x if x.startswith(args.data_dir) else os.path.join(args.data_dir, x) for x in info.audio_fp]
        assert all(os.path.exists(x) for x in extended_audio_fps)
        info.audio_fp = extended_audio_fps

        for annot_fp in extended_annot_fps:
            if annot_fp != "None":
                selection_table = pd.read_csv(annot_fp, delimiter = '\t')
                annots = list(selection_table['Annotation'].astype(str))
                all_annots.extend(annots)
        info.to_csv(info_fp)

    label_set = sorted(set(all_annots))
    label_mapping = {x : x for x in label_set}
    label_mapping['Unknown'] = 'Unknown'
    unknown_label = 'Unknown'

    if unknown_label in label_set:
        label_set.remove(unknown_label)

    setattr(args, "label_set", label_set)
    setattr(args, "label_mapping", label_mapping)
    setattr(args, "unknown_label", unknown_label)

    save_params(args)

if __name__ == "__main__":
    project_setup(sys.argv[1:])
