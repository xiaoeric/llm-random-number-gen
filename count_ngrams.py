import json
from nltk.util import ngrams
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import argparse


def count_ngrams_real(data, n=2):
    c = Counter()
    for num in data:
        decimal = str(num).split(".")[1]
        c.update(ngrams([*decimal], n))
    return c

def bigram_count_to_matrix(counts):
    mat = np.zeros((10, 10), dtype=int)
    for i in range(10):
        for j in range(10):
            mat[i, j] = counts[(str(i), str(j))]
    return mat

def plot_heatmap(mat, x_labels, y_labels, title=""):
    fig, ax = plt.subplots()
    im = ax.imshow(mat)

    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, mat[i, j],
                        ha="center", va="center", color="w")
            
    ax.set_title(title)
    # fig.tight_layout()


# bigram_counts = count_ngrams_real(data['chatgpt35_oneshot'])
# print(bigram_counts)
# # print(bigram_counts[('1', '1')])

# bigram_mat = bigram_count_to_matrix(bigram_counts)
# print(bigram_mat)

# digits = [str(i) for i in range(10)]
# plot_heatmap(bigram_mat, x_labels=digits, y_labels=digits, title="Counts of digit bigrams in real-numbers [0,1] (GPT-3.5 zeroshot)")
# plt.xlabel('second digit')
# plt.ylabel('first digit')
# plt.show()

def get_model_name(id, for_files=False):
    if 'gpt35' in id:
        return 'GPT-3.5' if not for_files else 'gpt35'
    if 'gpt4' in id:
        return 'GPT-4' if not for_files else 'gpt4'
    if 'llama_7b' in id:
        return 'LLaMA-7B' if not for_files else 'llama7b'
    if 'llama_13b' in id:
        return 'LLaMA-13B' if not for_files else 'llama13b'
    
def get_experiment_name(id, for_files=False):
    if 'oneshot' in id:
        return 'zero-shot' if not for_files else 'zero'
    if 'autoreg' in id:
        return 'conditional' if not for_files else 'cond'

def count_to_img(data_id):
    bigram_counts = count_ngrams_real(data[data_id])
    bigram_mat = bigram_count_to_matrix(bigram_counts)
    digits = [str(i) for i in range(10)]
    model = get_model_name(data_id)
    exp = get_experiment_name(data_id)
    plot_heatmap(bigram_mat, x_labels=digits, y_labels=digits, title=f'{model} {exp}')
    plt.xlabel('second digit')
    plt.ylabel('first digit')
    plt.savefig(f'figures/{get_model_name(data_id, for_files=True)}_{get_experiment_name(data_id, for_files=True)}_bigrams.png')

parser = argparse.ArgumentParser()
parser.add_argument("data_file")
# parser.add_argument("data_id")
# parser.add_argument("title")
# parser.add_argument("img_name")

args = parser.parse_args()

with open(args.data_file) as f:
    data = json.load(f)

for data_id in data:
    print("generating figure for " + data_id)
    count_to_img(data_id)

# bigram_counts = count_ngrams_real(data[args.data_id])
# bigram_mat = bigram_count_to_matrix(bigram_counts)
# digits = [str(i) for i in range(10)]
# plot_heatmap(bigram_mat, x_labels=digits, y_labels=digits, title=args.title)
# plt.xlabel('second digit')
# plt.ylabel('first digit')
# plt.savefig('figures/' + args.img_name)