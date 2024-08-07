# Voxaboxen

[![DOI](https://zenodo.org/badge/617502083.svg)](https://zenodo.org/badge/latestdoi/617502083)

Voxaboxen is a deep learning framework designed to find the start and stop times of (possibly overlapping) sound events in a recording. We designed it with bioacoustics applications in mind, so it accepts annotations in the form of [Raven](https://ravensoundsoftware.com/software/raven-lite/) selection tables.

If you use this software in your research, please [cite it](CITATION.cff).

![19_AL_Naranja_1025_detect](https://github.com/earthspecies/voxaboxen/assets/72874445/c69439c8-509b-4732-8d69-3bb38658ec9a)

## Installation

Requires `torch > 2.0` and the corresponding version of torchaudio. Other requirements can be installed with `pip install -r requirements.txt`.

Once cloned, Voxaboxen can be installed with `pip install -e .`.

## Quick start

Create a `train_info.csv` file with three columns:

- `fn`: Unique filename associated with each audio file
- `audio_fp`: Filepaths to audio files in train set
- `selection_table_fp`: Filepath to Raven selection tables

Repeat this for the other folds of your dataset, creating `val_info.csv` and `test_info.csv`. Run project setup and model training following the template in the Example Usage below.

Notes:
- Audio will be automatically resampled to 16000 Hz mono, no resampling is necessary prior to training.
- Selection tables are `.tsv` files. We only require the following columns: `Begin Time (s)`, `End Time (s)`, `Annotation`.

## Example usage:

Get the preprocessed [Meerkat (MT) dataset](https://zenodo.org/record/6012310):

`cd datasets/MT; wget https://storage.googleapis.com/esp-public-files/voxaboxen-demo/formatted.zip; unzip formatted.zip; wget https://storage.googleapis.com/esp-public-files/voxaboxen-demo/original_readme_and_license.md`

Project setup:

`python main.py project-setup --train-info-fp=datasets/MT/formatted/train_info.csv --val-info-fp=datasets/MT/formatted/val_info.csv --test-info-fp=datasets/MT/formatted/test_info.csv --project-dir=projects/MT_experiment`

Train model:

`python main.py train-model --project-config-fp=projects/MT_experiment/project_config.yaml --name=demo --lr=.00005 --batch-size=4`

Use trained model to infer annotations:

`python main.py inference --model-args-fp=projects/MT_experiment/demo/params.yaml --file-info-for-inference=datasets/MT/formatted/test_info.csv`

We provide a [Colab Notebook](https://colab.research.google.com/drive/1Qr1PQnw_bSUeXbvHSRuP91Pomxh1hfoi?usp=sharing) with more details about this process.

## Evaluation

We trained Voxaboxen on four bioacoustics datasets which represent a variety of animal species and recording conditions. These datasets were chosen because they contain expert annotations of bounding boxes that are precisely aligned with the onsets and offsets of vocalizations. In Table 1 we report the performance of Voxaboxen on a held-out test set from each of these datasets. 

As an informal baseline, we fine tuned an image-based [Faster-RCNN](https://papers.nips.cc/paper_files/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf) object detection model on each dataset. Adapted from the [Detectron2](https://github.com/facebookresearch/detectron2) code base, these networks were pre-trained with images in the COCO detection task and were fine tuned to detect vocalizations from spectrograms. 

For each of these experiments, we performed a small grid search to choose initial learning rate and batch size. For all four datasets, we found that Voxaboxen outperformed Faster-RCNN. 

| Dataset | Taxa | Num vox (train / val / test) | Num classes considered | F1@0.5 IoU Voxaboxen | F1@0.5 IoU Faster-RCNN |
| ------- | ---- | ---------------------------- | ---------------------- | -------------------- | ---------------------- |
| [BirdVox 10h](https://zenodo.org/record/6482837) | *Passeriformes* spp. | 4196 / 1064 / 3763 | 1 | 0.583 | 0.095 |
| [Meerkat](https://zenodo.org/record/6482837) | *Suricata suricatta* | 773 / 269 / 252 | 1 | 0.869 | 0.467 |
| [Powdermill](https://zenodo.org/record/4656848) | Bird spp. | 6849 / 2537 / 2854 | 10 | 0.447 | 0.250 |
| [Hawaii](https://zenodo.org/record/7078499) | Bird spp. | 24385 / 9937 / 18034 | 8 | 0.283 | 0.174 |

Table 1: Macro-averaged F1 score for each model, dataset pair. To compute these scores, we matched each predicted bounding box with at most one human-annotated bounding box, subject to the condition that the intersection over union (IoU) score of the proposed match was at least 0.5. Two of these datasets (BirdVox 10h and Meerkat) were previously used in the [DCASE few-shot detection task](https://dcase.community/challenge2022/task-few-shot-bioacoustic-event-detection). The code for dataset formatting can be found in [datasets](datasets) and the code for replicating these experiments can be found in [scripts](scripts).

## Editing Project Config

After running `python main.py project-setup`, a `project_config.yaml` file will be created in the project directory you specified. This config file codifies how different labels will be handled by any model within this project. This config file is automatically generated by the project setup script, but you can edit this file to revise how these labels are handled. There are a few things you can edit:

1. `label_set`: This is a list of all the label types that a model will be able to output. It is automatically populated with all the label types that appear in the `Annotation` column of the selection table. If you want your model to ignore a particular label type, perhaps because there are few events with that label type, you must delete that label type from this list.

2. `label_mapping`: This is a set of `key: value` pairs. Often, it is useful to group multiple types of labels into one. For example, maybe in your data there are multiple species from the same family, and you would like the model to treat this entire family with one label type. Upon training, Voxaboxen converts each annotation that appears as a `key` into the label specified by the corresponding `value`. When modifying `label_mapping`, you should ensure that each `value` that appears in `label_mapping` either also appears in `label_set`, or is the `unknown_label`.

3. `unknown_label`: This is set to `Unknown` by default. Any sound event labeled with the `unknown_label` will be treated as an event of interest, but the label type of the event will be treated as unknown. This may be desireable when there are vocalizations that are clearly audible, but are difficult for an annotator to identify to species. When the model is trained, it learns to predict a uniform distribution across possible label types whenever it encounters an event with the `unknown_label`. When the model is evaluated, it is not penalized for predicting the label of events which are annotated with the `unknown_label`. The `unknown_label` should not appear in the `label_set`.

For example, say you annotate your audio with the labels Red-eyed Vireo `REVI`, Philidelphia Vireo`PHVI`, and Unsure `REVI/PHVI`. To reflect your uncertainty about `REVI/PHVI`, your `label_set` would include `REVI` and `PHVI`, and your `label_mapping` would include the pairs `REVI: REVI`, `PHVI: PHVI`, and `REVI/PHVI: Unknown`. Alternatively, you could group both types of Vireo together by making your `label_set` only include `Vireo`, and your `label_mapping` include `REVI: Vireo`, `PHVI: Vireo`, `REVI/PHVI: Vireo`.

## Other features

Here are some additional options that can be applied during training:

- Flag `--stereo` accepts stereo audio. Order of channels matters; used for e.g. speaker diarization.
- Flag `--bidirectional` predicts the ends of events in addition to the beginning, matches starts and ends based on IoU. May improve box regression.
- Flag `--segmentation-based` switches to a frame-based approach. If used, we recommend putting `--rho=1`.
- Flag `--mixup` applies mixup augmentation.

## The name

Voxaboxen is designed to put a *box* around each vocalization (*vox*). It also rhymes with [Roxaboxen](https://www.thriftbooks.com/w/roxaboxen_alice-mclerran/331707/).
