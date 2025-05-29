# reproduce reported results on the existing bioacoustic SED datasets and the natural portion of OZF
for DATASET in AnSet BV_slowed HbW Katy_slowed MT OZF Pow; do
    uv run main.py project-setup --data-dir=datasets/${DATASET} --project-dir=projects/${DATASET}_experiment
    uv run main.py train-model --n-epochs=50 --batch-size=4 --encoder-type=beats --beats-checkpoint-fp=weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --bidirectional --project-config-fp=projects/${DATASET}_experiment/project_config.yaml --name=${DATASET}-reproduction
done

# reproduce reported results on the OZF synthetic datasets, which have varying overlap ratio
for overlap_ratio in 0 0.2 0.4 0.6 0.8 1; do
    uv run main.py project-setup --data-dir=datasets/OZF_synthetic/overlap_${overlap_ratio}_slowed --project-dir=projects/OZF_synth_${overlap_ratio}_slowed_experiment
    uv run main.py train-model --n-epochs=50 --batch-size=4 --encoder-type=beats --beats-checkpoint-fp=weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --bidirectional --project-config-fp=projects/OZF_synth_${overlap_ratio}_slowed_experiment/project_config.yaml --name=OZF_synth_${overlap_ratio}_slowed-reproduction
done
