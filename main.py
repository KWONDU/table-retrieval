import csv
import numpy as np
import os
import sys
from util import load_dataset, load_dataset_split, load_retrieval, parser


def main(dataset_name, dataset_splits, retrieval_method, table_elements, ks, **kwargs):
    # Load dataset
    source_dataset = load_dataset(dataset_name)
    print(source_dataset.dataset)

    # Retrieve tables
    for split in dataset_splits:
        dataset = load_dataset_split(source_dataset, split)
        if dataset == None:
            continue
        retrieval = load_retrieval(dataset_name, dataset, retrieval_method)
        print(f"### {retrieval_method.upper()} Retrieval - {dataset_name} {split} set ###")

        for target in table_elements:
            if not os.path.exists(f'retrieval_indices/{dataset_name.lower()}-{split}-{retrieval_method}-{target}.csv'):
                with open(f'retrieval_indices/{dataset_name.lower()}-{split}-{retrieval_method}-{target}.csv', 'w', newline='') as f:
                    sorted_indices = retrieval.retrieval(target=target)
                    writer = csv.writer(f)
                    writer.writerows(sorted_indices)
    
    # Index top-k tables
    for k in ks:
        k = int(k)
        print(f"### Score @{k} ###")
        
        result_list = []
        for split in dataset_splits:
            for target in table_elements:
                with open(f'retrieval_indices/{dataset_name.lower()}-{split}-{retrieval_method}-{target}.csv', 'r') as f:
                    reader = csv.reader(f)
                    top_k_indices = np.array(list(reader), dtype=int)[:, :k]
                    query_indices = np.arange(len(top_k_indices)).reshape(-1, 1)
                    recall_top_k = np.any(np.equal(top_k_indices, query_indices), axis=1)
                    recall_top_k_score = np.mean(recall_top_k) * 100
                    recall_flag = "exact match" if k == 1 else "recall"
                    
                    result = "{0:<15} serialization: {1:<10} | {2} score: {3:.2f}".format(f"[{split}]", target, recall_flag ,recall_top_k_score)
                    result_list.append(result + '\n')
                    print(result)

    # Save to file
    with open(f'results/{dataset_name.lower()}-{retrieval_method}-top-{k}.txt', 'w') as f:
        f.writelines(result_list)

    return


if __name__ == '__main__':
    parser = parser()
    args, _ = parser.parse_known_args()
    print(args)

    main(
        dataset_name=args.d,
        dataset_splits=args.sp.split(','),
        retrieval_method=args.rtr,
        table_elements=args.elem.split(','),
        ks=args.k.split(',')
        )
