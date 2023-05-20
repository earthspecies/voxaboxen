# sound_event_detection

Example usage:

Get the birdvox full night dataset (https://zenodo.org/record/1205569). Put the files in `datasets/birdvox_full_night/raw`. Run the script `process_birdvox_full_night.py`.

Get pretrained weights for aves. You will need to specify the default location of these weights in the files `source/training/params.py` and `source/active_learning/params.py`, or pass as a flag explicitly.

## Workflow:

1. `main.py project-setup ...flags...`. See `source/project/params.py`.
1.b Optional: edit project config to reflect label mapping preferences
2. `main.py active-learning-sampling ...flags...`. See `source/active_learning/params.py`.
3. annotate
4. `main.py train-model ...flags...`. See `source/training/params.py`.
5. Repeat 2-4 as desired.
6. `main.py inference ...flags...`. See `source/inference/params.py`.

Examples of this workflow are simulated in the different `scriptes/*experiment*.py` files. The flag `--query-oracle` replaces manual sampling with looking up predefined annotations. This is used purely for benchmarking the different active learning approaches. 