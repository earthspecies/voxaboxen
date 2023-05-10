# sound_event_detection

Example usage:

Get carrion crow data (the folder `19` and `Annotations_revised_by_Daniela.cleaned`) from GCP. Put them in the same directory.

Get pretrained weights for aves.

edit and run `make_detection_data_crows.py`

run
`python main.py --name=sweep3 --lr=0.001 --n-epochs=20 --clip-duration=16 --batch-size=8 --omit-empty-clip-prob=0.5 --clip-hop=8 --lamb=.02`

for active learning, run
`python active_learning_sampling.py ...other args...` Example arguments are in the file `active_learning_sampling.py`


New workflow, under main:
1. `project_setup`
2. `active_learning_sampling`
3. annotate or `query_oracle`
4. `active_learning_update`
5. `trian_model`
6. Repeat 2-5 as desired.

These are set up in `benchmarking`