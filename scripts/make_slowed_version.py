"""
Script to make slowed-down version of datasets, as in the associated paper
"""

import os
import shutil

import pandas as pd
import soundfile as sf

dataset_dirs = [
    "Katy",
    "BV",
    "OZF",
    os.path.join("OZF_synthetic", "overlap_0"),
    os.path.join("OZF_synthetic", "overlap_0.2"),
    os.path.join("OZF_synthetic", "overlap_0.4"),
    os.path.join("OZF_synthetic", "overlap_0.6"),
    os.path.join("OZF_synthetic", "overlap_0.8"),
    os.path.join("OZF_synthetic", "overlap_1"),
]

for dataset_dir in dataset_dirs:
    dataset_dir = os.path.join("datasets", dataset_dir)
    if not os.path.exists(dataset_dir):
        print(f"Dataset {dataset_dir} does not exist, skipping")

    print(f"Processing {dataset_dir}")
    new_dataset_dir = dataset_dir + "_slowed"

    if os.path.exists(new_dataset_dir):
        shutil.rmtree(new_dataset_dir)

    new_audio_dir = os.path.join(new_dataset_dir, "audio")
    new_st_dir = os.path.join(new_dataset_dir, "selection_tables")

    for d in [new_audio_dir, new_st_dir]:
        os.makedirs(d)

    if "Katy" in dataset_dir:
        scale_factor = 6
    else:
        scale_factor = 2

    for split in ["train", "val", "test"]:
        info_fp = os.path.join(dataset_dir, f"{split}_info.csv")
        info_df = pd.read_csv(info_fp)

        for _i, row in info_df.iterrows():
            audio, sr = sf.read(os.path.join(dataset_dir, row["audio_fp"]))
            new_sr = sr // scale_factor

            new_audio_fp = os.path.join(
                new_audio_dir, os.path.basename(row["audio_fp"])
            )
            sf.write(new_audio_fp, audio, new_sr)

            st = pd.read_csv(
                os.path.join(dataset_dir, row["selection_table_fp"]), sep="\t"
            )
            st["Begin Time (s)"] = st["Begin Time (s)"] * scale_factor
            st["End Time (s)"] = st["End Time (s)"] * scale_factor

            if "Low Freq (Hz)" in st.columns:
                st["Low Freq (Hz)"] = st["Low Freq (Hz)"] / scale_factor
            if "High Freq (Hz)" in st.columns:
                st["High Freq (Hz)"] = st["High Freq (Hz)"] / scale_factor

            new_st_fp = os.path.join(
                new_st_dir, os.path.basename(row["selection_table_fp"])
            )
            st.to_csv(new_st_fp, sep="\t", index=False)

        info_df.to_csv(os.path.join(new_dataset_dir, f"{split}_info.csv"), index=False)
