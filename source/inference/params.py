import argparse
import os
import yaml
from source.training.params import load_params
from pathlib import Path

def parse_inference_args(inference_args):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--model-args-fp', type=str, required=True, help = "filepath of model params saved as a yaml")
  parser.add_argument('--file-info-for-inference', type=str, required=True, help = "filepath of info csv listing filenames and filepaths of audio for inference")
  
  inference_args = parser.parse_args(inference_args)  
  return inference_args
