import argparse
import matplotlib.pyplot as plt
import re
import sys


datasets = ['FeTaQA', 'QTSumm']
ks = [1, 3, 5, 10, 20, 50]
pattern_em = r'\[(\w+)\]\s+serialization:\s+(\w+)\s+\|\s+exact match score:\s+([\d.]+)'
pattern_recall = pattern = r'\[(\w+)\]\s+serialization:\s+(\w+)\s+\|\s+recall score:\s+([\d.]+)'

def plot_scores(scores_info, dataset, split, method):
    plt.figure(figsize=(8, 8))
    for retrieval_range, scores in scores_info.items():
        plt.plot(ks, scores, label=retrieval_range, marker='o')
    plt.xlabel("Recall @K")
    plt.ylabel(f"{method.upper()} Score")
    plt.title(f"{dataset} {split.title()} {method.upper()} Scores")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'graphs/{dataset.lower()}-{split}-{method}.png')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-rtr', type=str, default='', help="retrieval method")
    args, unparsed = parser.parse_known_args()

    for dataset in datasets:
        result_list = {'train': {}, 'validation': {}, 'test': {}}
        for split, _ in result_list.items():
            result_list[split] = {'title': [], 'header': [], 'metadata': [], 'table': [], 'full': []}
        
        methods = {'sparse': 'bm25', 'dense': 'dpr'}
        if args.rtr not in methods:
            print("Invalid method!")
            sys.exit()

        for k in ks:
            pattern = pattern_em if k == 1 else pattern_recall
            with open(f'results/{dataset.lower()}-sparse-{methods[args.rtr]}-top-{k}.txt', 'r') as f:
                lines = f.readlines()

            for line in lines:
                match = re.search(pattern, line)
                if match:
                    split = match.group(1)
                    retrieval_range = match.group(2)
                    score = float(match.group(3))
        
                if split in result_list:
                    result_list[split][retrieval_range].append(score)
        
        for split, scores_info in result_list.items():
            plot_scores(scores_info, dataset, split, methods[args.rtr])
