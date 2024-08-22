import re
from util import parser, plot_scores


class MyDic():
    def __init__(self):
        self.data = {}
    
    def __iter__(self):
        for subject, labels in self.data.items():
            info = [(label, tuple(zip(*values))) for label, values in labels.items()]
            yield subject, info
    
    def append(self, subject, label, value):
        self.data[subject] = {} if subject not in self.data else self.data[subject]
        self.data[subject][label] = [] if label not in self.data[subject] else self.data[subject][label]
        self.data[subject][label].append(value)


if __name__ == '__main__':
    parser = parser()
    args, _ = parser.parse_known_args()
    dictionary = MyDic()

    pattern_em = r'\[(\w+)\]\s+serialization:\s+(\w+)\s+\|\s+exact match score:\s+([\d.]+)'
    pattern_recall = r'\[(\w+)\]\s+serialization:\s+(\w+)\s+\|\s+recall score:\s+([\d.]+)'
    dataset = args.d
    retrieval = args.rtr

    for k in args.k.split(','):
        k = int(k)
        
        try:
            with open(f'results/{dataset.lower()}-{retrieval}-top-{k}.txt', 'r') as file:
                lines = file.readlines()
        except FileNotFoundError as e:
            print(f"Invalid file: {dataset.lower()}-{retrieval}-top-{k}.txt")
            exit(1)
        
        # Extract components
        pattern = pattern_em if k == 1 else pattern_recall
        for line in lines:
            match = re.search(pattern, line)
            if match:
                split = match.group(1)
                target = match.group(2)
                score = float(match.group(3))

            dictionary.append(subject=split, label=target, value=(k, score))

    # Visualize
    for split, scores_info in dictionary:
        plot_scores((dataset, split, retrieval), scores_info)
