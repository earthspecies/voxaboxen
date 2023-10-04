# Using detectron2 for sound event detection

## Example usage

For training:
```
python main.py train-comparison --name=debug01 --n-epochs=200 --project-config-fp=/home/jupyter/sound_event_detection/projects/synthetic/project_config.yaml  --num-workers=0  --omit-empty-clip-prob=0.5  --batch-size=4 --opts MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.8 MODEL.WEIGHTS ""
```
This should also automatically run evaluation on the train, val and test set.

For evaluation/inference only:
```
cd source/comparisons
python evaluate.py --file-info-for-inference=/home/jupyter/sound_event_detection/datasets/synthetic/formatted/train_info.csv --full-param-fp=/home/jupyter/sound_event_detection/projects/synthetic/debug01/all_params.yaml --results-folder-name=train_results
```
Also see other options inside `evaluate.py`.

## Explanation of detectron params for training script

If there are redundant options in the detectron cfg and the sound event config, I attempted to use the sound event config. 

- `--detectron-base-config`: this is the base configuration file that will be loaded in first. To overwrite options with custom settings, use the subsequent arguments specified below.
  - Default: "./COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
  - List available in https://github.com/facebookresearch/detectron2/tree/57bdb21249d5418c130d54e2ebdc94dda7a4c01a/configs
  - If you do not want the pretrained model weights of this network included, include `--opts MODEL.WEIGHTS ""`
- `--detectron-use-box-statistics`: if you include this flag, the code will set the anchors based on the statistics of boxes in the training set
  - Otherwise it will use those in the default base configuration file.
  - Note that these custom anchors will not be the same as included in the default model, so will require new parts of the model to be trained (if you're using pretrained weights)
- `--detectron-config-fp`: path to yaml file with custom detectron cfg settings
  - If you want to change cfg.MODEL.ANCHOR_GENERATOR.SIZES or cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS, you should make a new yaml file and using this argument (instead of --ops below). This is because it is hard to specify list of lists as PATH.KEY value pairs in command line.
- `--opts`: specifying custom detectron cfg settings from the command line 
  - See an example above
  - See [here](https://github.com/facebookresearch/detectron2/blob/57bdb21249d5418c130d54e2ebdc94dda7a4c01a/docs/tutorials/configs.md) and [here](https://github.com/facebookresearch/detectron2/blob/57bdb21249d5418c130d54e2ebdc94dda7a4c01a/detectron2/engine/defaults.py#L134) for usage information
