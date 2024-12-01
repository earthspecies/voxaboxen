import os
from voxaboxen.evaluation.raven_utils import Clip
from voxaboxen.evaluation.evaluation import f1_from_counts
import voxaboxen.evaluation.metrics as metrics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


c = Clip(label_set=[1])
c.duration = 60.
annot_names = ['Benj', 'Logan', 'Louis']
labels_nums = [39, 137, 150, 174, 211]

def load_raven(fp):
    df = pd.read_csv(fp, sep='\t', index_col=0)
    df = df.loc[df['View']=='Spectrogram 1']
    df['Annotation'] = 1
    return df

def load_audacity(fp):
    df = pd.read_csv(fp, header=0, names=['Begin Time (s)', 'End Time (s)'], sep='\t', index_col=2)
    df.reset_index(drop=True, inplace=True)
    #df = df.drop('Unnamed: 2', axis=1)
    #df.columns = ['Selection', 'View', 'Channel', 'Low Freq (Hz)', 'High Freq (Hz)']
    df['Annotation'] = 1
    return df

def load(lable_num, annot_name):
    in_fp = f'overlapping_annots/ZF-common-labels{ln}-{annot_name}.txt'
    out_fp = f'overlapping_annots/ZF-common-labels{ln}-{annot_name}.csv'
    if annot_name=='Louis':
        df = load_audacity(in_fp)
    else:
        df = load_raven(in_fp)
    df.to_csv(out_fp, sep='\t')
    return out_fp


def overlaps_from_fp(df_fp):
    df = pd.read_csv(df_fp, sep='\t', index_col=0)
    overlap_nexts = df['End Time (s)'][:-1].array > df['Begin Time (s)'][1:].array
    overlaps = np.logical_or([False] + list(overlap_nexts), list(overlap_nexts) + [False])
    return df.loc[overlaps], np.argwhere(overlaps)

def display_dfs_from_ar(scores, scores_name):
    scores += scores.transpose((0,2,1))
    scores += np.tile(np.eye(len(annot_names)), (len(labels_nums),1,1))
    column_index = pd.MultiIndex.from_product([
        labels_nums,
        annot_names,
    ])

    # Reshape the data and create DataFrame
    # Reshape to (5, 9) to match the MultiIndex columns
    full_df = pd.DataFrame(
        scores.transpose(1, 0, 2).reshape(3, -1),
        columns=column_index,
        index=annot_names
    )
    full_df.to_csv(f'overlapping_annots/inter-annot-scores-{scores_name}.csv')
    print(full_df)
    mean_df = full_df.groupby(level=1, axis=1).mean()

    sns.heatmap(mean_df, annot=True, cmap='viridis', fmt='.4f', vmin=0.5, vmax=1.0)
    plt.gca().xaxis.tick_top()
    plt.title(scores_name + 'Scores')
    plt.savefig(heatmap_fp:=f'overlapping_annots/heatmap-means--{scores_name}.png')
    os.system(f'/usr/bin/xdg-open {heatmap_fp}')
    plt.clf()

def start_end_ar(df):
    return np.array(df[['Begin Time (s)', 'End Time (s)']]).T

def scores_from_ref_est(ref, est):
    ref, est = start_end_ar(ref), start_end_ar(est)
    #ref = np.array(ref[['Begin Time (s)', 'End Time (s)']]).T
    matching = metrics.match_events(ref, est, min_iou=0.5, method="fast")
    tp = len(matching)
    assert est.shape[0]==2
    fp = len(est[0]) - len(matching)
    fn = len(ref[0]) - len(matching)
    assert fp >= 0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2*tp / (2*tp + fp + fn)
    return f1, prec, rec

def return_dense_chunks(ref_df, est_df):
    ref, est = start_end_ar(ref_df), start_end_ar(est_df)
    start = min(ref.min(), est.min())
    end = max(ref.max(), est.max())
    split_points = np.arange(start, end+1.999, 2)
    ns_in_chunk = []
    refs_chunk_idxs = []
    ests_chunk_idxs = []
    for i, chunk_start in enumerate(split_points[:-1]):
        chunk_end = min(split_points[i+1], end)
        refs_in_chunk = np.logical_and(ref[1] > chunk_start, ref[0] < chunk_end)
        ests_in_chunk = np.logical_and(est[1] > chunk_start, est[0] < chunk_end)
        refs_chunk_idxs.append(refs_in_chunk)
        ests_chunk_idxs.append(ests_in_chunk)
        ns_in_chunk.append(refs_in_chunk.sum() + ests_in_chunk.sum())

    ns_in_chunk = np.array(ns_in_chunk)
    assert ( ns_in_chunk.sum() >= ref.shape[1] + est.shape[1]) # some in >1 chunk
    print(ns_in_chunk)
    k = len(ns_in_chunk)//10
    assert k > 0
    thresh = ns_in_chunk[np.argsort(ns_in_chunk)[-k:]].min()
    dense_idxs = ns_in_chunk > thresh
    ref_idxs = np.array(refs_chunk_idxs)[dense_idxs].astype(int).sum(axis=0) > 0
    est_idxs = np.array(ests_chunk_idxs)[dense_idxs].astype(int).sum(axis=0) > 0
    return ref_df.loc[ref_idxs], est_df.loc[est_idxs]


fullscores = np.zeros([len(labels_nums), len(annot_names), len(annot_names)])
oscores = np.zeros([len(labels_nums), len(annot_names), len(annot_names)])
dscores = np.zeros([len(labels_nums), len(annot_names), len(annot_names)])
for i, ln in enumerate(labels_nums):
    for j, annot1_name in enumerate(annot_names):
        annot1_fp = load(ln, annot1_name)
        full_ref = pd.read_csv(annot1_fp, sep='\t', index_col=0)
        c.load_annotations(annot1_fp)
        for k, annot2_name in enumerate(annot_names[j+1:]):
        #for k, annot2_name in enumerate(annot_names):
            annot2_fp = load(ln, annot2_name)
            full_est = pd.read_csv(annot2_fp, sep='\t', index_col=0)
            c.load_predictions(annot2_fp)
            c.compute_matching(IoU_minimum=0.5)
            raw_counts = c.evaluate()[1]
            f1dict = f1_from_counts(raw_counts['TP'], raw_counts['FP'], raw_counts['FN'])
            fullscores[i, j, j+k+1] = f1dict['f1']
            print(annot1_name, annot2_name, f1dict)

            overlap_ref, overlaps1 = overlaps_from_fp(annot1_fp)
            overlap_est, overlaps2 = overlaps_from_fp(annot2_fp)
            prec = scores_from_ref_est(full_ref, overlap_est)[1]
            rec = scores_from_ref_est(full_est, overlap_ref)[1]
            print(prec, rec)
            oscores[i, j, j+k+1] = 2*prec*rec / (prec+rec)

            dense_ref, dense_est = return_dense_chunks(full_ref, full_est)
            f1, prec, rec = scores_from_ref_est(dense_ref, dense_est)
            dscores[i, j, j+k+1] = f1
            #scores[i, j, k] = f1dict['f1']

display_dfs_from_ar(fullscores, 'full')
display_dfs_from_ar(oscores, 'overlapping')
display_dfs_from_ar(dscores, 'dense_sections')
