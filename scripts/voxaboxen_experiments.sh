cd ..

MODEL=hubert;


# lr=$1
# gpu=$2
# export CUDA_VISIBLE_DEVICES=$gpu
# for dset in Anuraset BV_slowed hawaii humpback katydids_slowed MT powdermill OZF; do
#         echo python main.py train-model --n-epochs=50 --batch-size=4 --encoder-type=beats --beats-checkpoint-fp=gcp-weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --num-workers=2 --project-config-fp=projects/${dset}_experiment/project_config.yaml --name=${dset}-${lr}-beats --lr ${lr} --overwrite
#         taskset -c 0-30 python main.py train-model --n-epochs=50 --batch-size=4 --encoder-type=beats --beats-checkpoint-fp=gcp-weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --num-workers=2 --project-config-fp=projects/${dset}_experiment/project_config.yaml --name=${dset}-${lr}-beats --lr ${lr} --overwrite
# done

for DATASET in MT;
do
python main.py project-setup --train-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/train_info.csv --val-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/val_info.csv --test-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/test_info.csv --project-dir=projects/${DATASET}_${MODEL}_experiment
for lr in .00001;
do
python main.py train-model --project-config-fp=projects/${DATASET}_${MODEL}_experiment/project_config.yaml --name=${MODEL}_${lr} --lr=${lr} --batch-size=4 --scale-factor=320 --clip-duration=10 --bidirectional --encoder-type=$MODEL
done
done

for DATASET in OZF_slowed_0.5 BV_slowed_0.5 Anuraset humpback katydids_slowed_0.16666666666666666 powdermill hawaii;
do
python main.py project-setup --train-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/train_info.csv --val-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/val_info.csv --test-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/test_info.csv --project-dir=projects/${DATASET}_${MODEL}_experiment
for lr in .00003;
do
python main.py train-model --project-config-fp=projects/${DATASET}_${MODEL}_experiment/project_config.yaml --name=${MODEL}_${lr} --lr=${lr} --batch-size=4 --scale-factor=320 --clip-duration=10 --bidirectional --encoder-type=$MODEL
done
done







# for DATASET in overlap_0.2_slowed_0.5 overlap_0.4_slowed_0.5 overlap_0.6_slowed_0.5 overlap_0.8_slowed_0.5 overlap_1_slowed_0.5;
# do
# python main.py project-setup --train-info-fp=/home/jupyter/data/voxaboxen_data/OZF_synthetic/${DATASET}/train_info.csv --val-info-fp=/home/jupyter/data/voxaboxen_data/OZF_synthetic/${DATASET}/val_info.csv --test-info-fp=/home/jupyter/data/voxaboxen_data/OZF_synthetic/${DATASET}/test_info.csv --project-dir=projects/${DATASET}_${MODEL}_experiment
# for lr in .00001;
# do
# python main.py train-model --project-config-fp=projects/${DATASET}_${MODEL}_experiment/project_config.yaml --name=${MODEL}_${lr} --lr=${lr} --batch-size=8 --scale-factor=320 --clip-duration=10 --rho=1 --segmentation-based --encoder-type=$MODEL
# done
# done