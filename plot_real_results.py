import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad

with open('chatgpt.json') as f:
    chatgpt_data = json.load(f)
with open('llama.json') as f:
    llama_data = json.load(f)

def make_boxplot(samples):
    fig = plt.figure(figsize=(12, 5))
    plt.boxplot(samples, labels=['uniform',
                                'GPT-3.5\nNARS',
                                'GPT-3.5\nARS',
                                'GPT-4\nNARS',
                                'GPT-4\nARS',
                                'LLaMa-7B\nNARS',
                                'LLaMa-7B\nARS',
                                'LLaMa-13B\nNARS',
                                'LLaMa-13B\nARS',])
    
paired_colors = mpl.colormaps['Paired']
cmap = {
    'uniform': 'black',
    'GPT-3.5': '#00A67E',
    'GPT-4': '#a96bf9',
    'LLaMA-7B': '#a7c0dd',
    'LLaMA-13B': '#0472f2',
}

def make_stacked_hist(samples, labels, condition):
    fig = plt.figure(figsize=(8, 8))
    bins = np.linspace(0.0, 1.0, 21)
            
    for i, label in enumerate(labels):
        plt.hist(samples[i], bins, density=True, label=label, color=cmap[label], histtype='step', stacked=True)
    
    plt.legend()
    plt.ylabel('density')
    plt.xlabel('value')
    plt.title(f'{condition} sampling over real numbers on [0, 1]')


def pdf_plot(samples, labels, condition):
    x = np.linspace(0, 1, 41)
            
    for i, label in enumerate(labels):
        density = stats.gaussian_kde(samples[i])
        def adjusted_density(x):
            return density(x) / uniform_density(x)
        plt.plot(x, adjusted_density(x) / quad(adjusted_density, 0, 1)[0], label=label, color=cmap[label])
    
    plt.legend()
    plt.ylabel('prob. density')
    plt.xlabel('value')
    plt.ylim(bottom=0)
    plt.title(f'{condition} sampling over real numbers on [0, 1]')


uniform = np.linspace(0, 1, 100)
uniform_density = stats.gaussian_kde(uniform)

samples = [uniform,
           np.array(chatgpt_data['chatgpt35_oneshot']),
           np.array(chatgpt_data['chatgpt35_autoreg']),
           np.array(chatgpt_data['chatgpt4_oneshot']),
           np.array(chatgpt_data['chatgpt4_autoreg']),
           np.array(llama_data['llama_7b_oneshot']),
           np.array(llama_data['llama_7b_autoreg']),
           np.array(llama_data['llama_13b_oneshot']),
           np.array(llama_data['llama_13b_autoreg']),]

labels = ['uniform',
          'GPT-3.5',
          'GPT-4',
          'LLaMA-7B',
          'LLaMA-13B',]

zeroshot_samples = [uniform,
                    np.array(chatgpt_data['chatgpt35_oneshot']),
                    np.array(chatgpt_data['chatgpt4_oneshot']),
                    np.array(llama_data['llama_7b_oneshot']),
                    np.array(llama_data['llama_13b_oneshot']),]

conditional_samples = [uniform,
                       np.array(chatgpt_data['chatgpt35_autoreg']),
                       np.array(chatgpt_data['chatgpt4_autoreg']),
                       np.array(llama_data['llama_7b_autoreg']),
                       np.array(llama_data['llama_13b_autoreg']),]

# make_stacked_hist(zeroshot_samples, labels, 'Zero-shot')
pdf_plot(zeroshot_samples, labels, 'Zero-shot')
plt.show()


# make_stacked_hist(conditional_samples, labels, 'Conditional')
pdf_plot(conditional_samples, labels, 'Conditional')
plt.show()