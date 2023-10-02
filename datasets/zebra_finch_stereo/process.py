import os 
import argparse
import subprocess
import sklearn
from sklearn import model_selection
import pandas as pd 
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--splits", type=str, nargs="+", help="Splits to write in csv (train, val)"
)

BUCKET = "gs://zebra_finch_interactive_playbacks/"

def download_txt(args):
    command = "gsutil cp -R " + BUCKET 
    if args.splits[0] == 'train':
        command += "train_data/"
    else:
        command += "test_data/"
    command += " . "
    try:
        result=subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    except Exception as e:
        print("Failed to download file from {}!".format(BUCKET))
        raise e

def download_wav(gs_path, local_path, args):
    command = "gsutil cp "
    command += BUCKET + gs_path + " " + local_path
    try:
        result=subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    except Exception as e:
        print("Failed to download file from {}!".format(BUCKET))
        raise e
    filename = result.stderr.split("\n")[0].split("gs://")[-1].replace("...", "").split("/")[-1]
    return os.path.join(local_path, filename)

def convert_to_stereo(local_path1, local_path2, out_path):
    command = "ffmpeg -y -i " + local_path1 + " -i " + local_path2
    command += " -filter_complex '[0:a][1:a]amerge' -ac 2 -ar 16000 " + out_path
    try:
        result=subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    except Exception as e:
        print("Failed to mix files")
        raise e


def main(args):
    
    download_txt(args)
    if args.splits[0] == 'train':
        subset_path = os.path.join(os.path.abspath(os.getcwd()),'train_data')
    else:
        subset_path = os.path.join(os.path.abspath(os.getcwd()),'test_data')
    raven_files = [f for f in os.listdir(os.path.join(subset_path,'annotations')) if f.endswith(".txt")]
    assert len(raven_files) > 0, "No Raven files found in {}".format(subset_path)
    os.makedirs(os.path.join(subset_path, "wav"), exist_ok=True)
    os.makedirs(os.path.join(subset_path, "tmp"), exist_ok=True)
    
    if len(args.splits) > 0:
        splits = sklearn.model_selection.train_test_split(raven_files, test_size=0.1, random_state=42)
        splits = dict(zip(args.splits, splits))
    else:
        splits = dict(zip(args.splits, [raven_files]))

    for split, split_files in splits.items():
        md = pd.DataFrame(columns=['fn','audio_fp', 'selection_table_fp'])
        for i,f in enumerate(tqdm.tqdm(split_files)):
            parts = f.split(".")[0].split("_",1)
            fnames = []
            for c in ['Left','Right']:
                gs_path = parts[0] + "/" + c + "/" + '*' + "_" + parts[1] + ".wav"
                fnames.append(download_wav(gs_path, os.path.join(subset_path, "tmp"), args))
            audio_fp = os.path.join(subset_path, "wav", parts[0] + '_' + parts[1] + ".wav")
            convert_to_stereo(fnames[0], fnames[1], audio_fp)
            md.loc[i] = [os.path.basename(audio_fp), audio_fp, os.path.join(subset_path, 'annotations', f)]
        md.to_csv(os.path.join(os.path.dirname(subset_path), split+".csv"), index=False)
        
        
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    
'''
python process.py --splits test
creates
'test_data'
containing the annotations and the corresponding wav files for both pairs
and
test.csv formatted according to the voxaboxen requirements
'''