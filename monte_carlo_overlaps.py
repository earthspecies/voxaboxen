import numpy as np

d = 0.1
n = 20

sims = []
preds = []
lengths = d*np.random.dirichlet(np.ones(n))
#lengths = np.ones(n)*d/n
#pred2 = np.triu(np.expand_dims(lengths, 1) + lengths, 1).sum()
assert np.allclose(lengths.sum(), d)
overlaps = []
for i in range(number:=100000):
    np.random.shuffle(lengths)
    onsets = np.sort(np.random.rand(n))
    ends = onsets+lengths
    true_d = d - max(0, ends[-1]-1)
    pred = true_d*(n-1)
    #if true_d!=d:
        #print(i, ends)
    preds.append(pred)
    #noverlaps = (ends[:-1] > onsets[1:]).sum()
    overlaps.append(np.triu(np.expand_dims(ends[:-1], 1) > onsets[1:]))
    noverlaps = np.triu(np.expand_dims(ends[:-1], 1) > onsets[1:]).sum()
    sims.append(noverlaps)

predicted = d*(n-1)
sims = np.array(sims)
overlaps = np.stack(overlaps).mean(axis=0)
pred = np.array(preds).mean()
means = np.stack(sims).mean()
print(f'predicted inf birds: {pred:.4f} predicted: {pred*7/8:.4f} mean: {sims.mean():.4f}, std: {sims.std()/number**0.5:.4f}')

