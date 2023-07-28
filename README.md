# sound_event_detection

Requires `torch > 2.0` and the corresponding version of torchaudio. Other requirements can be installed with `pip intall -r requirements.txt`. To use audio in formats such as `mp3` and `flac` you may need to `apt install ffmpeg` (or `conda install ffmpeg` if using conda).

## Example usage:

Get the birdvox full night dataset (https://zenodo.org/record/1205569). Put the files in `datasets/birdvox_full_night/raw`. Run the script `process_birdvox_full_night.py`.

Get pretrained weights for AVES, which is the backbone model. (https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.pt) Put them in `weights`. 

Process data into correct format:

`cd datasets/birdvox_full_night; python process_birdvox_full_night.py; cd .. ; cd ..`

Project setup:

`python main.py project-setup --train-info-fp=/home/jupyter/sound_event_detection/datasets/birdvox_full_night/formatted/train_info.csv --test-info-fp=/home/jupyter/sound_event_detection/datasets/birdvox_full_night/formatted/test_info.csv --project-dir=/home/jupyter/sound_event_detection/projects/bvfn`

Train model:

`python main.py train-model --project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn/project_config.yaml --name=demo --clip-duration=2 --clip-hop=1`

## FAQ

Q: How is this different from BirdNet?

A: BirdNet does multi-label classification on a window of audio. This model detects each vocalization individually, so you can have multiple detections in a short window.

Q: How is this different from TweetyNet?

A: TweetyNet doesn't allow for overlapping vocalizations, which are common in the wild.