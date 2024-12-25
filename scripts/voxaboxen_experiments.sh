cd ..

MODEL=frame_atst;

for DATASET in MT Anuraset hawaii humpback powdermill;
do
python main.py project-setup --train-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/train_info.csv --val-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/val_info.csv --test-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/test_info.csv --project-dir=projects/${DATASET}_${MODEL}_experiment
for lr in .00001 .00003 .0001 .0003 .001;
do
python main.py train-model --project-config-fp=projects/${DATASET}_${MODEL}_experiment/project_config.yaml --name=${MODEL}_${lr} --lr=${lr} --batch-size=8 --scale-factor=640 --clip-duration=10 --rho=1 --segmentation-based --encoder-type=$MODEL
done
done
