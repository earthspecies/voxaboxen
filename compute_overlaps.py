import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def stats_from_section(df):
    df = df.sort_values('Begin Time (s)')
    maybe_emptys = df.loc[df['End Time (s)'] <= df['Begin Time (s)']]
    if len(maybe_emptys) > 0:
        print(fn)
        with open('empty_boxes.txt', 'a') as f:
            f.write(fn + '\n')
        print(maybe_emptys)
    df = df.loc[df['End Time (s)'] > df['Begin Time (s)']]
    nboxes = df.shape[0]
    onsets, ends = df['Begin Time (s)'].to_numpy(), df['End Time (s)'].to_numpy()
    boxlens = ends - onsets
    noverlaps = np.triu(np.expand_dims(ends[:-1], 1) > onsets[1:]).sum()
    if (boxlens==0).any():
        breakpoint()
    return nboxes, noverlaps, boxlens

def split_into_segs(df, seglen):
    split_points = np.arange(0, df['End Time (s)'].max(), seglen)
    segments = []
    for i, sp in enumerate(split_points[:-1]):
        next_sp = split_points[i+1]
        seg_df = df.loc[(df['Begin Time (s)'] >= sp) & (df['End Time (s)'] < next_sp)]
        segments.append(seg_df)
    return segments

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset_name', '-d', type=str, default='OZF')
    parser.add_argument('--show-plots', action='store_true')
    ARGS = parser.parse_args()

    filewise_nboxes = []
    filewise_noverlaps = []
    expected_filewise_noverlaps = []
    filewise_boxlens = []
    segwise_nboxes = []
    segwise_noverlaps = []
    expected_segwise_noverlaps = []
    segwise_boxlens = []
    ds = []
    if ARGS.dset_name=='OZF':
        st_dir = 'ZFdset'
        out_dir= 'ZFdset/analysis'
    else:
        st_dir = f'datasets/{ARGS.dset_name}/formatted/selection_tables'
        os.makedirs(out_dir:=f'datasets/{ARGS.dset_name}/analysis', exist_ok=True)
    for fn in os.listdir(st_dir):
        fp = os.path.join(st_dir, fn)
        if os.path.isdir(fp):
            continue
        if ARGS.dset_name=='hawaii' and 'Recording' in fp:
            continue
        df = pd.read_csv(fp, sep='\t')
        if 'View' in df.columns:
            df = df.loc[df['View']=='Spectrogram 1']
        nbxs, nols, bxls = stats_from_section(df)
        if nbxs==0:
            continue
        if nols > 40:
            print(fn)
        filewise_nboxes.append(nbxs); filewise_noverlaps.append(nols)
        filewise_boxlens += list(bxls)
        d = sum(bxls) / 60
        ds.append(d)
        expected_noverlaps = d*(nbxs-1)*(1 - (1/8) - (1/nbxs))
        expected_filewise_noverlaps.append( expected_noverlaps)
        file_seg_noverlaps = []
        for seg_df in split_into_segs(df, 10):
            nbxs_seg, nols_seg, bxls_seg = stats_from_section(seg_df)
            segwise_nboxes.append(nbxs_seg); segwise_noverlaps.append(nols_seg)
            file_seg_noverlaps.append(nols_seg)
            segwise_boxlens += list(bxls_seg)
            expected_segwise_noverlaps.append((nbxs_seg-1) * sum(bxls_seg) / 10)

    def make_hist(data, dname, binsize):
        plt.hist(data, bins=np.arange(min(data), max(data), binsize))
        title = ' '.join([w[0].upper() + w[1:] for w in dname.split('_')])
        plt.title(title, fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(hist_path:=f'{out_dir}/{dname}-histogram.png')
        if ARGS.show_plots:
            os.system(f'/usr/bin/xdg-open {hist_path}')
        plt.clf()

    print(f'total files with no overlap: {(np.array(filewise_noverlaps)==0).sum()}')
    os.makedirs('out_dir', exist_ok=True)
    make_hist(filewise_nboxes, 'num_vocs_per_file', 20)
    make_hist(filewise_noverlaps, 'num_overlaps_per_file', 2)
    make_hist(filewise_boxlens, 'vocs_lengths', .01)
    plt.scatter(filewise_nboxes, filewise_noverlaps)
    plt.xlabel('Number of Vocs', fontsize=14)
    plt.ylabel('Num Overlapping Vocs', fontsize=14)
    plt.title('Num Vocs vs Overlaps per 60s Window', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(fwise_scatter_fp:=f'{out_dir}/filewise-nvocs-vs-noverlaps.png')
    if ARGS.show_plots:
        os.system(f'/usr/bin/xdg-open {fwise_scatter_fp}')
    plt.clf()

    make_hist(segwise_nboxes, 'num_vocs_per_10s_segment', 3)
    make_hist(segwise_noverlaps, 'num_overlaps_per_10s_segment', 1)
    plt.scatter(segwise_nboxes, segwise_noverlaps)
    plt.xlabel('Number of Vocs', fontsize=13)
    plt.ylabel('Num Overlapping Vocs', fontsize=13)
    plt.title('Num Vocs vs Overlaps per 10s Window', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(swise_scatter_fp:=f'{out_dir}/segwise-nvocs-vs-noverlaps.png')
    if ARGS.show_plots:
        os.system(f'/usr/bin/xdg-open {swise_scatter_fp}')

    print(f'Total Boxes: {sum(filewise_nboxes)}')
    print(f'Total Overlapping Boxes: {sum(filewise_noverlaps)}')
    print(f'Overlap ratio: {sum(filewise_noverlaps)/sum(filewise_nboxes):.4f}')
    print(f'Boxes per File Range: {min(filewise_nboxes)} - {max(filewise_nboxes)}')
    print(f'Noverlaps per File Range: {min(filewise_noverlaps)} - {max(filewise_noverlaps)}')
    print(f'Boxes per 10s Segment: {min(segwise_nboxes)} - {max(segwise_nboxes)}')
    print(f'Noverlaps per 10s Segment: {min(segwise_noverlaps)} - {max(segwise_noverlaps)}')
    fd = np.array(ds)
    fn = np.array(filewise_nboxes)
    fo = np.array(filewise_noverlaps)
    efo = np.array(expected_filewise_noverlaps)
    diff = efo-fo
    df = pd.DataFrame({'file': np.arange(len(fn)), 'n':fn, 'd':fd, 'B':8, 'expected overlaps':efo, 'observed overlaps':fo, 'difference':diff})
    print(f'Diff mean: {diff.mean():.5f}, Diff Std: {diff.std():.5f}')
    df.to_latex(f'{out_dir}/significance-table.tex', index=False, float_format="%.2f")

