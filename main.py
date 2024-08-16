import argparse
import csv
import numpy as np
import os
import sys


def main(dataset_name, retrieval_method, k, **kwargs):
    if dataset_name == 'FeTaQA':
        from dataset import FeTaQADataset
        dataset = FeTaQADataset()
    elif dataset_name == 'QTSumm':
        from dataset import QTSummDataset
        dataset = QTSummDataset()
    else:
        print("Invalid dataset!")
        sys.exit()
    print(dataset.dataset)
    
    methods = {'sparse': 'bm25', 'dense': 'dpr'}
    splits = ['train', 'validation', 'test']
    range_of_retrieval = ['title', 'header', 'metadata', 'table', 'full']
    for split in splits:
        retrieval_dataset = dataset.dataset[split]
        print(f"### {retrieval_method.title()} Retrieval - {methods[retrieval_method].upper()} | {dataset_name} dataset - {split} ###")
        if retrieval_method == 'sparse':
            from retrieval import SparseRetrieval
            retrieval = SparseRetrieval(dataset_name=dataset_name, dataset=retrieval_dataset)
            for retrieval_range in range_of_retrieval:
                if not os.path.exists(f'retrieval_indices/{dataset_name.lower()}-{split}-{methods[retrieval_method]}-{retrieval_range}.csv'):
                    with open(f'retrieval_indices/{dataset_name.lower()}-{split}-{methods[retrieval_method]}-{retrieval_range}.csv', 'w', newline='') as f:
                        sorted_indices = retrieval.bm25_retrieval(retrieval_range=retrieval_range)
                        writer = csv.writer(f)
                        writer.writerows(sorted_indices.tolist())
        elif retrieval_method == 'dense':
            from retrieval import DenseRetrieval
            retrieval = DenseRetrieval(dataset_name=dataset_name, dataset=retrieval_dataset)
            for retrieval_range in range_of_retrieval:
                if not os.path.exists(f'retrieval_indices/{dataset_name.lower()}-{split}-{methods[retrieval_method]}-{retrieval_range}.csv'):
                    with open(f'retrieval_indices/{dataset_name.lower()}-{split}-{methods[retrieval_method]}-{retrieval_range}.csv', 'w', newline='') as f:
                        sorted_indices = retrieval.dpr_retrieval(retrieval_range=retrieval_range)
                        writer = csv.writer(f)
                        writer.writerows(sorted_indices)
        else:
            print("Invalid method!")
            sys.exit()

    save_to_file = []
    print(f"### Score @{k} ###")
    for split in splits:
        for retrieval_range in range_of_retrieval:
            with open(f'retrieval_indices/{dataset_name.lower()}-{split}-{methods[retrieval_method]}-{retrieval_range}.csv', 'r') as f:
                reader = csv.reader(f)
                top_k_indices = np.array(list(reader), dtype=int)[:, :k]
                query_indices = np.arange(len(top_k_indices)).reshape(-1, 1)
                recall_top_k = np.any(np.equal(top_k_indices, query_indices), axis=1)
                recall_top_k_score = np.mean(recall_top_k) * 100
                recall_flag = "exact match" if k == 1 else "recall"
                result = "{0:<15} serialization: {1:<10} | {2} score: {3:.2f}".format(f"[{split}]", retrieval_range, recall_flag ,recall_top_k_score)
                save_to_file.append(result + '\n')
                print(result)

    with open(f'results/{dataset_name.lower()}-{retrieval_method}-{methods[retrieval_method]}-top-{k}.txt', 'w') as f:
        f.writelines(save_to_file)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='', help="dataset name")
    parser.add_argument('-rtr', type=str, default='', help="retrieval method")
    parser.add_argument('-k', type=int, default=1, help="top-k")
    args, unparsed = parser.parse_known_args()
    print(args)

    main(dataset_name=args.d, retrieval_method=args.rtr, k=args.k)
