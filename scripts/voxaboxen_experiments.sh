
# reproduce reported results on the existing bioacoustic SED datasets and the natural portion of OZF
for DATASET in Anuraset BV_slowed hawaii katydids_slowed MT powdermill OZF_slowed; do
    python main.py project-setup --train-info-fp=datasets/${DATASET}/formatted/train_info.csv --val-info-fp=datasets/${DATASET}/formatted/val_info.csv --test-info-fp=datasets/${DATASET}/formatted/test_info.csv --project-dir=projects/${DATASET}_experiment
    python main.py train-model --n-epochs=50 --batch-size=4 --encoder-type=beats --beats-checkpoint-fp=weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --bidirectional --project-config-fp=projects/${DATASET}_experiment/project_config.yaml --name=${DATASET}-reproduction
done

# reproduce reported results on the OZF synthetic datasets, which have varying overlap ratio
for overlap_ratio in 0 0.2 0.4 0.6 0.8 1; do
    python main.py project-setup --train-info-fp=datasets/OZF_synth_${overlap_ratio}_slowed/formatted/train_info.csv --val-info-fp=datasets/OZF_synth_${overlap_ratio}_slowed/formatted/val_info.csv --test-info-fp=datasets/OZF_synth_${overlap_ratio}_slowed/formatted/test_info.csv --project-dir=projects/OZF_synth_${overlap_ratio}_slowed_experiment
    python main.py train-model --n-epochs=50 --batch-size=4 --encoder-type=beats --beats-checkpoint-fp=weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --bidirectional --project-config-fp=projects/OZF_synth_${overlap_ratio}_slowed_experiment/project_config.yaml --name=OZF_synth_${overlap_ratio}_slowed-reproduction
done

