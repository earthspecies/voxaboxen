import argparse
import os
import yaml
from voxaboxen.training.params import load_params

def parse_inference_args(inference_args):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--model-args-fp', type=str, required=True, help = "filepath of model params saved as a yaml")
  parser.add_argument('--file-info-for-inference', type=str, required=True, help = "filepath of info csv listing filenames and filepaths of audio for inference")
  parser.add_argument('--detection-threshold', type=float, default=0.5, help="detection peaks need to be at or above this threshold to make it into the exported selection table")
  parser.add_argument('--classification-threshold', type=float, default=0.0, help="classification probability needs to be at or above this threshold to not be labeled as Unknown")
  
  inference_args = parser.parse_args(inference_args)  
  return inference_args
