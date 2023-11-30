import json
import numpy as np
from scipy import stats


def samples_to_bins(samples, n):
    bins = [int(s * n) for s in samples]
    return bins


def total_variation_dist(samples, n):
    bins = samples_to_bins(samples, n)
    print(bins[:10])
    unique, counts = np.unique(bins, return_counts=True)
    prob_dict = dict(zip(unique, counts / len(bins)))
    tvd = 0
    uniform = 1 / n
    for i in range(n):
        if i in unique:
            prob = prob_dict[i]
        else:
            prob = 0
        tvd += abs(prob - uniform)
    return tvd / 2


with open('llama.json') as f:
    data = json.load(f)

print(data.keys())
samples = np.array(data['llama_13b_autoreg'])

print(len(samples))

avg = np.average(samples)
med = np.median(samples)
std = np.std(samples)
mode = stats.mode(samples)
    # # chi_2 = chi_square(n)
tvd = total_variation_dist(samples, 100)
    

    # print(f'stats for n={n}, count={len(samples)}')
print('mean:\t', avg)
print('median:\t', med)
print('std:\t', std)
print('mode:\t', mode)
    # # print('chi_2:\t', chi_2)
print('tvd:\t', tvd)