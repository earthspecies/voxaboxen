import os
import subprocess
import librosa
import pandas as pd
from tqdm import tqdm

# Define the LaTeX table content

datasets = ["Anuraset", "BV", "hawaii", "humpback", "katydids", "MT", "OZF", "powdermill"]
dataset_names = [x.replace('_', ' ') for x in datasets]
formatted_dataset_parent_dir = "/home/jupyter/data/voxaboxen_data/"

columns = ["Dataset", "N files", "N Classes", "Duration (hr)", "N events", "Avg. event dur. (sec)", "Recording type","Location", "Taxa"]

tex = r"""
\documentclass{article}
\usepackage{booktabs}
\begin{document}

\begin{table}[h]
    \centering
    \begin{tabular}{||"""

tex+= f"{len(columns)*'c|'}"

tex += r"""|}
        \toprule
        
"""

for i, column in enumerate(columns):
    tex += f"{column}"
    if i < len(columns)-1:
        tex += " & "
    else:
        tex += r""" \\
        \midrule
        
        """
        
for dataset_name, dataset in sorted(zip(dataset_names, datasets)):
    data_dir = os.path.join(formatted_dataset_parent_dir, dataset, "formatted")
    splits = ["train", "val", "test"]
    manifest = []
    for split in splits:
        manifest.append(pd.read_csv(os.path.join(data_dir, f"{split}_info.csv")))
    manifest = pd.concat(manifest)
    
    total_dur = 0
    n_events = 0
    total_event_dur = 0
    classes = []
    
    for i, row in tqdm(manifest.iterrows(), total=len(manifest)):
        audio_fp = os.path.join(row["audio_fp"])
        total_dur += librosa.get_duration(path=audio_fp, sr=None) / 3600
        
        st_fp = os.path.join(row["selection_table_fp"])
        st = pd.read_csv(st_fp, sep='\t')
        st["Duration"] = st["End Time (s)"] - st["Begin Time (s)"]
        st = st[st["Annotation"] != "Unknown"]
        total_event_dur += st["Duration"].sum()
        classes += list(st["Annotation"].unique())
        
        n_events += len(st)
    
    for i, column in enumerate(columns):
        if column == "Dataset":
            entry = dataset_name
        elif column == "N files":
            entry = len(manifest)
        elif column == "Duration (hr)":
            entry = '%.2f'%(total_dur)
        elif column == "N events":
            entry = n_events
        elif column == "Recording type":
            dd = {"Anuraset" : "Terrestrial PAM", "BV" : "Terrestrial PAM", "hawaii" : "Terrestrial PAM", "humpback" : "Underwater PAM", "katydids" : "Terrestrial PAM", "MT" : "On-body", "OZF" : "Laboratory", "powdermill" : "Terrestrial PAM"}
            
            entry = dd[dataset]
        elif column == "Taxa":
            dd = {"Anuraset" : "Anura", "BV" : "Passeriformes", "hawaii" : "Aves", "humpback" : "Megaptera novaeangliae", "katydids" : "Tettigoniidae", "MT" : "Suricata suricatta", "OZF" : "Taeniopygia castanotis", "powdermill" : "Passeriformes"}
            
            entry = dd[dataset]
            
        
        elif column == "Location":
            dd = {"Anuraset" : "Brazil", "BV" : "New York, USA", "hawaii" : "Hawaii, USA", "humpback" : "North Pacific Ocean", "katydids" : "Panam\'a", "MT" : "South Africa", "OZF" : "Laboratory", "powdermill" : "Pennsylvania, USA"}
            
            entry = dd[dataset]
            
        elif column == "Avg. event dur. (sec)":
            entry = '%.2f'%(total_event_dur / n_events)
            
            
        
        elif column == "N Classes":
            entry = len(set(classes))
        
        else:
            entry = " "
            
        tex += f"{entry}"
        if i < len(columns)-1:
            tex += " & "
        else:
            tex += r""" \\
            \midrule

            """
    
    
#     print(f"Processing {dataset}")
    
#     
    
#     

        
tex += r"""
        \bottomrule
    \end{tabular}
    \caption{Sample Table}
    \label{tab:sample}
\end{table}
\end{document}
"""

target_directory = formatted_dataset_parent_dir

# Ensure the target directory exists
os.makedirs(target_directory, exist_ok=True)

# Define the LaTeX source file path and PDF file path
latex_file_path = os.path.join(target_directory, 'metadata.tex')
pdf_file_path = os.path.join(target_directory, 'metadata.pdf')

# Write the LaTeX code to a file
with open(latex_file_path, 'w') as f:
    f.write(tex)

# Compile the LaTeX file to PDF
try:
    subprocess.run(['pdflatex', '-output-directory', target_directory, latex_file_path], check=True)
    print(f"PDF successfully created at {pdf_file_path}")
except subprocess.CalledProcessError as e:
    print(f"Error compiling LaTeX: {e}")
