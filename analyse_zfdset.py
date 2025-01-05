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
    noverlaps1 = (np.expand_dims(ends[:-1], 1) > onsets[1:]).diagonal().sum()
    assert noverlaps >= noverlaps1
    #if noverlaps > noverlaps1:
        #print(888)
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
    filewise_nboxes = []
    filewise_noverlaps = []
    expected_filewise_noverlaps = []
    filewise_boxlens = []
    segwise_nboxes = []
    segwise_noverlaps = []
    expected_segwise_noverlaps = []
    segwise_boxlens = []
    for fn in os.listdir('ZFdset'):
        fp = os.path.join('ZFdset', fn)
        if os.path.isdir(fp):
            continue
        df = pd.read_csv(fp, sep='\t', index_col=0)
        df = df.loc[df['View']=='Spectrogram 1']
        nbxs, nols, bxls = stats_from_section(df)
        if nols==0:
            print(f'no overlaps in {fn}')
            if nbxs > 230:
                breakpoint()
        elif nols > 80:
            print(fn)
        filewise_nboxes.append(nbxs); filewise_noverlaps.append(nols)
        filewise_boxlens += list(bxls)
        expected_filewise_noverlaps.append((nbxs-1) * sum(bxls) / 60)
        file_seg_noverlaps = []
        for seg_df in split_into_segs(df, 10):
            nbxs_seg, nols_seg, bxls_seg = stats_from_section(seg_df)
            segwise_nboxes.append(nbxs_seg); segwise_noverlaps.append(nols_seg)
            file_seg_noverlaps.append(nols_seg)
            segwise_boxlens += list(bxls_seg)
            if nols_seg > 17:
                breakpoint()
            expected_segwise_noverlaps.append((nbxs_seg-1) * sum(bxls_seg) / 10)

    def make_hist(data, dname, binsize):
        plt.hist(data, bins=np.arange(min(data), max(data), binsize))
        title = ' '.join([w[0].upper() + w[1:] for w in dname.split('_')])
        plt.title(title)
        plt.savefig(hist_path:=f'ZFdset/analysis/{dname}-histogram.png')
        os.system(f'/usr/bin/xdg-open {hist_path}')
        plt.clf()

    print(f'total files with no overlap: {(np.array(filewise_noverlaps)==0).sum()}')
    os.makedirs('ZFdset/analysis', exist_ok=True)
    make_hist(filewise_nboxes, 'num_vocs_per_file', 20)
    make_hist(filewise_noverlaps, 'num_overlaps_per_file', 2)
    make_hist(filewise_boxlens, 'vocs_lengths', .01)
    plt.scatter(filewise_nboxes, filewise_noverlaps)
    plt.xlabel('Number of Vocs')
    plt.ylabel('Num Overlapping Vocs')
    plt.title('Num Vocs vs Num Overlaps per 60s Window')
    plt.savefig(fwise_scatter_fp:='ZFdset/analysis/filewise-nvocs-vs-noverlaps.png')
    os.system(f'/usr/bin/xdg-open {fwise_scatter_fp}')
    plt.clf()

    make_hist(segwise_nboxes, 'num_vocs_per_10s_segment', 3)
    make_hist(segwise_noverlaps, 'num_overlaps_per_10s_segment', 1)
    #make_hist(segwise_boxlens, 'segwise_boxlens', .05)
    plt.scatter(segwise_nboxes, segwise_noverlaps)
    plt.xlabel('Number of Vocs')
    plt.ylabel('Num Overlapping Vocs')
    plt.title('Num Vocs vs Num Overlaps per 10s Window')
    plt.savefig(swise_scatter_fp:='ZFdset/analysis/segwise-nboxes-vs-noverlaps.png')
    os.system(f'/usr/bin/xdg-open {swise_scatter_fp}')

    print(f'Total Boxes: {sum(filewise_nboxes)}')
    print(f'Total Overlapping Boxes: {sum(filewise_noverlaps)}')
    print(f'Boxes per File Range: {min(filewise_nboxes)} - {max(filewise_nboxes)}')
    print(f'Noverlaps per File Range: {min(filewise_noverlaps)} - {max(filewise_noverlaps)}')
    print(f'Boxes per 10s Segment: {min(segwise_nboxes)} - {max(segwise_nboxes)}')
    print(f'Noverlaps per 10s Segment: {min(segwise_noverlaps)} - {max(segwise_noverlaps)}')
    fn = np.array(filewise_nboxes)
    fn = np.array(filewise_noverlaps)
    fd = fn * 0.1/60
    breakpoint()

