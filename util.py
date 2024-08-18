import argparse
import importlib
import matplotlib.pyplot as plt


def load_dataset(dataset_name):
    try:
        module = importlib.import_module('dataset')
        source_dataset = getattr(module, f'{dataset_name}Dataset')()
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Invalid dataset: {dataset_name}")
        sys.exit()
    print(source_dataset.dataset)

    return source_dataset


def load_dataset_split(source_dataset, split):
    try:
        dataset = getattr(source_dataset, split)
    except AttributeError as e:
        print(f"Invalid dataset split: {split}")
        sys.exit()
    
    return dataset


def load_retrieval(dataset_name, dataset, retrieval_method):
    try:
        module = importlib.import_module('retrieval')
        retrieval = getattr(module, f'{retrieval_method.upper()}Retrieval')(dataset_name=dataset_name, dataset=dataset)
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Invalid retrieval method: {retrieval_method}")
        sys.exit()
    
    return retrieval


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='', help="dataset name")
    parser.add_argument('-sp', type=str, default='', help="dataset splits")
    parser.add_argument('-rtr', type=str, default='', help="retrieval method")
    parser.add_argument('-elem', type=str, default='', help="tabular data elements")
    parser.add_argument('-k', type=str, default='', help="top-k tables")

    return parser


def plot_scores(title_triple, scores_info):
    dataset, split, retrieval = title_triple

    plt.figure(figsize=(8, 8))
    for target, (ks, scores) in scores_info:
        plt.plot(list(ks), list(scores), label=target, marker='o')
    plt.xlabel("Recall @K")
    plt.ylabel(f"{retrieval.upper()} Score")
    plt.title(f"{dataset} {split.title()} {retrieval.upper()} Scores")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'graphs/{dataset.lower()}-{split}-{retrieval}.png')
    plt.close()

    return
