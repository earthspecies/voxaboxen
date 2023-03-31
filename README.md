# sound_event_detection

Example usage:

Get carrion crow data (the folder `19` and `Annotations_revised_by_Daniela.cleaned`) from GCP. Put them in the same directory.

edit and run `make_detection_data.all_crows.py`

run
`python main.py --name=unfreeze --lr=0.0001 --n-epochs=100 --clip-duration=10 --batch-size=30 --pos-weight=10`
