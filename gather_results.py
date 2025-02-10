import yaml
import pandas as pd

all_results = {}
all_nobid_results = {}
for dset in ('Anuraset', 'BV_slowed', 'hawaii', 'katydids_slowed', 'MT', 'powdermill', 'OZF_slowed'):
    best_val_map = 0
    for lr in ('1e-4', '3e-5', '1e-5'):
        try:
            with open(f'projects/{dset}_experiment/{dset}-{lr}-beats-bid-goodeval/val_results.yaml') as f:
                vr = yaml.load(f, yaml.SafeLoader())
        except FileNotFoundError:
            print('no results file for', dset, lr)
            continue
        if vr['mAP@0.5'] > best_val_map:
            with open(f'projects/{dset}_experiment/{dset}-{lr}-beats-bid-goodeval/test_results.yaml') as f:
                all_results['dset'] = yaml.load(f, yaml.SafeLoader())
            with open(f'projects/{dset}_experiment/{dset}-{lr}-beats/test_results.yaml') as f:
                all_nobid_results['dset'] = yaml.load(f, yaml.SafeLoader())

df = pd.DataFrame(all_results).T
df = df.drop(['macro-f1@0.2', 'micro-f1@0.2'], axis=1)
df.to_csv('voxaboxen-beats-bid-goodeval-results.csv')
print('Bid')
print(df)

df_nobid = pd.DataFrame(all_results).T
df_nobid = df_nobid.drop(['macro-f1@0.2', 'micro-f1@0.2'], axis=1)
df_nobid.to_csv('voxaboxen-beats-nobid-goodeval-results.csv')
print('Nobid')
print(df_nobid)

