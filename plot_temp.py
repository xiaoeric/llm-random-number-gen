import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.integrate import quad

uniform = np.linspace(0, 1, 100)
uniform_density = stats.gaussian_kde(uniform)
cmap = {
    'uniform': 'black',
    'GPT-3.5': '#00A67E',
    'GPT-4': '#a96bf9',
    'LLaMA-7B': '#a7c0dd',
    'LLaMA-13B': '#0472f2',
}

def pdf_plot(samples, labels, cmap, title=""):
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
    plt.title(title)


def calc_tvd(samples, labels):
    for i, label in enumerate(labels):
        density = stats.gaussian_kde(samples[i])
        def adjusted_density(x):
            return density(x) / uniform_density(x)
        area = quad(adjusted_density, 0, 1)[0]
        def pdf(x):
            return adjusted_density(x) / area
        def abs_diff(x):
            return np.abs(pdf(x) - 1)
        tvd, err = quad(abs_diff, 0, 1)
        tvd /= 2
        print(f'{label}: {tvd}')


def get_samples_by_name(samples, name):
    samples_with_model = {}
    for key in samples:
        if name in key:
            samples_with_model[key] = samples[key]
    return samples_with_model


def make_title(samp='', model=''):
    title = f'{samp} sampling over real numbers on [0, 1]'
    if model != '':
        title += ', ' + model
    return title


samples = {}

with open('chatgpt.json') as f:
    chatgpt_data = json.load(f)
with open('llama.json') as f:
    llama_data = json.load(f)

samples['gpt35_zero_temp08'] = np.array(chatgpt_data['chatgpt35_oneshot'])
samples['gpt35_cond_temp08'] = np.array(chatgpt_data['chatgpt35_autoreg'])
samples['gpt4_zero_temp08'] = np.array(chatgpt_data['chatgpt4_oneshot'])
samples['gpt4_cond_temp08'] = np.array(chatgpt_data['chatgpt4_autoreg'])

samples['llama7b_zero_temp08'] = np.array(llama_data['llama_7b_oneshot'])
samples['llama7b_cond_temp08'] = np.array(llama_data['llama_7b_autoreg'])
samples['llama13b_zero_temp08'] = np.array(llama_data['llama_13b_oneshot'])
samples['llama13b_cond_temp08'] = np.array(llama_data['llama_13b_autoreg'])

with open('chatgpt_with_temperature.json') as f:
    chatgpt_temp_data = json.load(f)
with open('llama_with_temperature.json') as f:
    llama_temp_data = json.load(f)

print(chatgpt_temp_data.keys())

samples['gpt35_zero_temp12'] = np.array(chatgpt_temp_data['01_chatgpt35_temp12_results_oneshot.txt'])
samples['gpt35_zero_temp15'] = np.array(chatgpt_temp_data['01_chatgpt35_temp15_results_oneshot.txt'])
samples['gpt4_zero_temp12'] = np.array(chatgpt_temp_data['01_chatgpt4_temp12_results_oneshot.txt'])
samples['gpt4_zero_temp15'] = np.array(chatgpt_temp_data['01_chatgpt4_temp15_results_oneshot.txt'])
samples['gpt35_cond_temp12'] = np.array(chatgpt_temp_data['01_chatgpt35_temp12_results_autoreg.txt'])
samples['gpt35_cond_temp15'] = np.array(chatgpt_temp_data['01_chatgpt35_temp15_results_autoreg.txt'])
samples['gpt4_cond_temp12'] = np.array(chatgpt_temp_data['01_chatgpt4_temp12_results_autoreg.txt'])
samples['gpt4_cond_temp15'] = np.array(chatgpt_temp_data['01_chatgpt4_temp15_results_autoreg.txt'])

samples['llama7b_zero_temp12'] = np.array(llama_temp_data['01_llama7b_temp12_results_oneshot.txt'])
samples['llama7b_zero_temp15'] = np.array(llama_temp_data['01_llama7b_temp15_results_oneshot.txt'])
samples['llama13b_zero_temp12'] = np.array(llama_temp_data['01_llama13b_temp12_results_oneshot.txt'])
samples['llama13b_zero_temp15'] = np.array(llama_temp_data['01_llama13b_temp15_results_oneshot.txt'])
samples['llama7b_cond_temp12'] = np.array(llama_temp_data['01_llama7b_temp12_results_autoreg.txt'])
samples['llama7b_cond_temp15'] = np.array(llama_temp_data['01_llama7b_temp15_results_autoreg.txt'])
samples['llama13b_cond_temp12'] = np.array(llama_temp_data['01_llama13b_temp12_results_autoreg.txt'])
samples['llama13b_cond_temp15'] = np.array(llama_temp_data['01_llama13b_temp15_results_autoreg.txt'])


llama7b_zero_samples = [uniform] + list(get_samples_by_name(samples, 'llama7b_zero').values())
llama7b_cond_samples = [uniform] + list(get_samples_by_name(samples, 'llama7b_cond').values())
llama7b_temp_cmap = {
    'uniform': 'black',
    't=0.8': '#a7c0dd',
    't=1.2': '#a7dda9',
    't=1.5': '#ddc4a7',
}
llama13b_zero_samples = [uniform] + list(get_samples_by_name(samples, 'llama13b_zero').values())
llama13b_cond_samples = [uniform] + list(get_samples_by_name(samples, 'llama13b_cond').values())
llama13b_temp_cmap = {
    'uniform': 'black',
    't=0.8': '#0472f2',
    't=1.2': '#04f20d',
    't=1.5': '#f28404',
}


gpt35_zero_samples = [uniform] + list(get_samples_by_name(samples, 'gpt35_zero').values())
gpt35_cond_samples = [uniform] + list(get_samples_by_name(samples, 'gpt35_cond').values())
gpt35_temp_cmap = {
    'uniform': 'black',
    'default': '#00A67E',
    't=1.2': '#7ba600',
    't=1.5': '#a60028'
}
gpt4_zero_samples = [uniform] + list(get_samples_by_name(samples, 'gpt4_zero').values())
gpt4_cond_samples = [uniform] + list(get_samples_by_name(samples, 'gpt4_cond').values())
gpt4_temp_cmap = {
    'uniform': 'black',
    'default': '#a96bf9',
    't=1.2': '#6bf9f0',
    't=1.5': '#bbf96b'
}

def plot_llama():
    pdf_plot(llama7b_zero_samples, llama7b_temp_cmap.keys(), llama7b_temp_cmap, make_title('Zero-shot', 'LLaMA-7B'))
    plt.savefig('figures/llama7b_zero_temp.png')
    plt.show()
    
    pdf_plot(llama13b_zero_samples, llama13b_temp_cmap.keys(), llama13b_temp_cmap, make_title('Zero-shot', 'LLaMA-13B'))
    plt.savefig('figures/llama13b_zero_temp.png')
    plt.show()
    
    pdf_plot(llama7b_cond_samples, llama7b_temp_cmap.keys(), llama7b_temp_cmap, make_title('Conditional', 'LLaMA-7B'))
    plt.savefig('figures/llama7b_cond_temp.png')
    plt.show()

    pdf_plot(llama13b_cond_samples, llama13b_temp_cmap.keys(), llama13b_temp_cmap, make_title('Conditional', 'LLaMA-13B'))
    plt.savefig('figures/llama13b_cond_temp.png')
    plt.show()

def plot_chatgpt():
    pdf_plot(gpt35_zero_samples, gpt35_temp_cmap.keys(), gpt35_temp_cmap, make_title('Zero-shot', 'GPT-3.5'))
    plt.savefig('figures/gpt35_zero_temp.png')
    plt.show()

    pdf_plot(gpt4_zero_samples, gpt4_temp_cmap.keys(), gpt4_temp_cmap, make_title('Zero-shot', 'GPT-4'))
    plt.savefig('figures/gpt4_zero_temp.png')
    plt.show()

    pdf_plot(gpt35_cond_samples, gpt35_temp_cmap.keys(), gpt35_temp_cmap, make_title('Conditional', 'GPT-3.5'))
    plt.savefig('figures/gpt35_cond_temp.png')
    plt.show()

    pdf_plot(gpt4_cond_samples, gpt4_temp_cmap.keys(), gpt4_temp_cmap, make_title('Conditional', 'GPT-4'))
    plt.savefig('figures/gpt4_cond_temp.png')
    plt.show()

print('gpt4 zero')
calc_tvd(gpt4_zero_samples, gpt4_temp_cmap.keys())