
dset_name=$1

echo python main.py project-setup --train-info-fp=datasets/${1}/formatted/train_info.csv --val-info-fp=datasets/${1}/formatted/val_info.csv --test-info-fp=datasets/${1}/formatted/test_info.csv --project-dir=projects/${1}_experiment
python main.py project-setup --train-info-fp=datasets/${1}/formatted/train_info.csv --val-info-fp=datasets/${1}/formatted/val_info.csv --test-info-fp=datasets/${1}/formatted/test_info.csv --project-dir=projects/${1}_experiment
