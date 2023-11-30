import json
import numpy as np
import matplotlib.pyplot as plt

with open('chatgpt.json') as f:
    chatgpt_data = json.load(f)
with open('llama.json') as f:
    llama_data = json.load(f)

uniform = np.linspace(0, 1, 100)

samples = [uniform,
           np.array(chatgpt_data['chatgpt35_oneshot']),
           np.array(chatgpt_data['chatgpt35_autoreg']),
           np.array(chatgpt_data['chatgpt4_oneshot']),
           np.array(chatgpt_data['chatgpt4_autoreg']),
           np.array(llama_data['llama_7b_oneshot']),
           np.array(llama_data['llama_7b_autoreg']),
           np.array(llama_data['llama_13b_oneshot']),
           np.array(llama_data['llama_13b_autoreg']),]

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
plt.show()