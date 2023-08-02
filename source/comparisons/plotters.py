import sys
import json
import matplotlib.pyplot as plt

def plot_total_loss(fp):
    with open(fp, "r") as f:
        t = f.readlines()
    d = [json.loads(s[:-1]) for s in t]
    x = [_d["iteration"] for _d in d[10:] if "total_loss" in _d.keys()]
    y = [_d["total_loss"] for _d in d[10:] if "total_loss" in _d.keys()]
    plt.scatter(x, y)
    plt.savefig(fp.replace(".json","total_loss.png"))
    plt.close()


if __name__ == "__main__":
    fp = sys.argv[1]
    plot_total_loss(fp)