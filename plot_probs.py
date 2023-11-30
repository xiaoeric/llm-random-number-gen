from plot_utils import *
import pickle
from scipy.special import logsumexp
import numpy as np
import re

def load_pkl(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--domain')
    # parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--type', default='oneshot')
    parser.add_argument('--prompt', type=int, default=0)
    parser.add_argument('--model')
    parser.add_argument('--trial', type=int, default=-1)
    # parser.add_argument('--n-columns', type=int, default=2)

    args = parser.parse_args()

    if args.trial == -1:
        path = os.path.join(args.data_dir, args.domain, args.type, f'prompt-{args.prompt}', args.model)
        dir_list = os.listdir(path)
        counter = 0
        sum_probs = None
        size = None
        for dir in dir_list:
            if re.match(r'^trial-\d+$', dir):
                counter += 1
                data_path = os.path.join(path, dir, 'pcfg-logprobs.pkl')
                data = load_pkl(data_path)
                pred_logpobs = np.array(data[1])
                if sum_probs is None:
                    sum_probs = np.zeros_like(pred_logpobs)
                    size = len(pred_logpobs)
                norm_probs = np.exp(pred_logpobs - logsumexp(pred_logpobs))
                sum_probs += norm_probs
        avg_probs = sum_probs / counter
        fig = plt.figure(figsize=(8, 5))
        plt.bar(range(1,size+1), avg_probs)
        plt.axhline(y = 0.1, color = 'grey', linestyle = 'dashed')   
        plt.ylabel('Sampling Prob.')
        plt.xlabel('Generated Number')
        plt.title(f'LLaMa-7B, {args.prompt} Examples, NARS (n={counter})')
        fig.tight_layout()
        plt.show()
    else:
        data_path = os.path.join(args.data_dir, args.domain, args.type, f'prompt-{args.prompt}', args.model, f'trial-{args.trial}', 'pcfg-logprobs.pkl')
        data = load_pkl(data_path)
        pred_logpobs = np.array(data[1])

        norm_probs = np.exp(pred_logpobs - logsumexp(pred_logpobs))

        plt.bar(range(1,11), norm_probs)
        plt.ylabel('Sampling Prob.')
        plt.xlabel('Generated Number')
        plt.title(f'LLaMa-7B, {args.prompt} Examples, NARS')
        plt.show()
    


if __name__ == '__main__':
    main()