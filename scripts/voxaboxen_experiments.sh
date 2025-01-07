cd ..

MODEL=crnn;

for DATASET in MT;
do
python main.py project-setup --train-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/train_info.csv --val-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/val_info.csv --test-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/test_info.csv --project-dir=projects/${DATASET}_${MODEL}_experiment
for lr in .00001 .00003 .0001 .0003;
do
for hs in 64 128;
do
python main.py train-model --project-config-fp=projects/${DATASET}_${MODEL}_experiment/project_config.yaml --name=${MODEL}_${lr}_${hs} --lr=${lr} --batch-size=8 --scale-factor=320 --clip-duration=10 --rho=1 --segmentation-based --encoder-type=$MODEL --rnn-hidden-size=${hs} --overwrite
done
done
done