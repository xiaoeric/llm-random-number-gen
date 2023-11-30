import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def print_stats(n):
    samples = np.array(data[str(n)])
    norm_samples = samples / n
    avg = np.average(norm_samples)
    med = np.median(norm_samples)
    std = np.std(norm_samples)
    mode = stats.mode(samples)
    # chi_2 = chi_square(n)
    tvd = total_variation_dist(n)
    

    print(f'stats for n={n}, count={len(samples)}')
    print('mean:\t', avg)
    print('median:\t', med)
    print('std:\t', std)
    print('mode:\t', mode)
    # print('chi_2:\t', chi_2)
    print('tvd:\t', tvd)


def total_variation_dist(n):
    samples = np.array(data[str(n)])
    unique, counts = np.unique(samples, return_counts=True)
    prob_dict = dict(zip(unique, counts / len(samples)))
    tvd = 0
    uniform = 1 / n
    for i in range(1,n+1):
        if i in unique:
            prob = prob_dict[i]
        else:
            prob = 0
        tvd += abs(prob - uniform)
    return tvd / 2


def chi_square(n):
    samples = np.array(data[str(n)])
    unique, counts = np.unique(samples, return_counts=True)
    count_dict = dict(zip(unique, counts))
    chi_2 = 0
    expected = len(samples) / n
    for i in range(1,n+1):
        if i in unique:
            count = count_dict[i]
        else:
            count = 0
        chi_2 += ((count - expected) ** 2) / expected
    return chi_2


with open('chatgpt_results.json') as f:
    data = json.load(f)

for n in [10, 100, 1000, 10000, 100000]:
    print_stats(n)

uniform = np.linspace(0, 1, 200)
data_2d = np.stack([uniform] + [np.array(data[str(n)]) / n for n in [10, 100, 1000, 10000, 100000]])
plt.boxplot(data_2d.T, labels=['uniform'] + [f'n={n}' for n in [10, 100, 1000, 10000, 100000]])
plt.ylabel('')
plt.show()