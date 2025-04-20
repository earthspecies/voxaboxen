import os
import pandas as pd

data = {}
for fname in os.listdir("inference_times"):
    dataset, device, bid = fname.replace(".txt", "").split("-")
    bid = 'bid' if bid.endswith('True') else 'nobid'
    with open(os.path.join("inference_times", fname)) as f:
        val = float(f.read().strip())
    data[(device, bid)] = data.get((device, bid), {})
    data[(device, bid)][dataset] = val

df = pd.DataFrame(data)#.sort_index()
testset_durs = pd.Series({
    'Anuraset': 5.37,
    'BV_slowed': 2.00,
    'hawaii': 10.35,
    'humpback': 2.69,
    'katydids_slowed': 1.00,
    'MT': 0.25,
    'powdermill': 1.58,
    'OZF_slowed': 0.22
})

testset_durs *= 3600

df.index.name = "dataset"
df = df[[('cuda', 'nobid'), ('cuda', 'bid'), ('cpu', 'nobid'), ('cpu', 'bid')]]
for x,y in df.columns:
    df[(x, y+'-rtf')] = df[(x,y)] / testset_durs
print(df)
breakpoint()

latex = df.to_latex('inference_times.tex',
    float_format="%.3f",
    multicolumn=True,
    multicolumn_format='c'
)

