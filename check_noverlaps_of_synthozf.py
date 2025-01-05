import os
import pandas as pd
import numpy as np
from analyse_zfdset import stats_from_section

#for nov in np.linspace(0, 1, 6):
for nov in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    dset_info = pd.read_csv(f'datasets/OZF_synthetic/overlap_{nov}/data_info.csv')
    selection_table_fps = ['datasets' + x.removeprefix('/home/jupyter/data/voxaboxen_data') for x in dset_info['selection_table_fp']]
    tot_noverlaps, tot_nboxes = [], []
    for stfp in selection_table_fps:
        df = pd.read_csv(stfp, sep='\t')
        nboxes, noverlaps, boxlens = stats_from_section(df)
        tot_noverlaps.append(noverlaps)
        tot_nboxes.append(nboxes)

    tot_noverlaps = np.array(tot_noverlaps)
    tot_nboxes = np.array(tot_nboxes)
    #print(ratios:=tot_noverlaps/tot_nboxes)
    ratios = tot_noverlaps/tot_nboxes
    novmean = tot_noverlaps.mean()
    nbmean = tot_nboxes.mean()
    print(f'Mean noverlaps: {novmean:.3f}, Mean nboxes: {nbmean.mean():.3f}, Ratio of means: {novmean/nbmean:.3f}, Mean of ratios: {ratios.mean():.3f}')

