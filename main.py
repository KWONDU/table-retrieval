import csv
import json
import numpy as np
import os
from util import load_dataset, load_dataset_split, load_retrieval, parser


def main(dataset_name, dataset_splits, retrieval_method, table_elements, ks, **kwargs):
    # Load dataset
    source_dataset = load_dataset(dataset_name)
    # print(source_dataset.dataset)
    
    # Retrieve tables
    for split in dataset_splits:
        dataset = load_dataset_split(source_dataset, split)
        if dataset == None:
            continue
        retrieval = load_retrieval(dataset_name, dataset, retrieval_method)
        print(f"### {retrieval_method.upper()} Retrieval - {dataset_name} {split} set ###")

        if dataset_name == 'OpenWikiTable':
            if not os.path.exists(f'retrieval_indices/{dataset_name.lower()}-{split}-tid.json'):
                with open(f'retrieval_indices/{dataset_name.lower()}-{split}-tid.json', 'w') as file:
                    json.dump(dataset['tid'], file)

        for target in table_elements:
            if not os.path.exists(f'retrieval_indices/{dataset_name.lower()}-{split}-{retrieval_method}-{target}.csv'):
                with open(f'retrieval_indices/{dataset_name.lower()}-{split}-{retrieval_method}-{target}.csv', 'w', newline='') as file:
                    sorted_indices = retrieval.retrieval(target=target)
                    writer = csv.writer(file)
                    writer.writerows(sorted_indices)
    
    # Index top-k tables
    for k in ks:
        k = int(k)
        print(f"### Score @{k} ###")
        
        result_list = []
        for split in dataset_splits:
            for target in table_elements:
                with open(f'retrieval_indices/{dataset_name.lower()}-{split}-{retrieval_method}-{target}.csv', 'r') as file:
                    reader = csv.reader(file)

                    if dataset_name == 'OpenWikiTable':
                        with open(f'retrieval_indices/{dataset_name.lower()}-{split}-tid.json', 'r') as file:
                            idx_tid_map = json.load(file)
                        dup_sorted_tids = [[idx_tid_map[idx] for idx in row] for row in np.array(list(reader), dtype=int)]
                        sorted_tids = [list(dict.fromkeys(row)) for row in dup_sorted_tids]
                        top_k_tids = np.array(sorted_tids)[:, :k]
                        gold_tids = np.array(idx_tid_map).reshape(-1, 1)
                        recall_top_k = np.any(np.equal(top_k_tids, gold_tids), axis=1)
                    else:
                        top_k_indices = np.array(list(reader), dtype=int)[:, :k]
                        query_indices = np.arange(len(top_k_indices)).reshape(-1, 1)
                        recall_top_k = np.any(np.equal(top_k_indices, query_indices), axis=1)
                    recall_top_k_score = np.mean(recall_top_k) * 100
                    recall_flag = "exact match" if k == 1 else "recall"
                    
                    result = "{0:<15} serialization: {1:<10} | {2} score: {3:.2f}".format(f"[{split}]", target, recall_flag ,recall_top_k_score)
                    result_list.append(result + '\n')
                    print(result)

        # Save to file
        with open(f'results/{dataset_name.lower()}-{retrieval_method}-top-{k}.txt', 'w') as file:
            file.writelines(result_list)

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
