#!/bin/bash

# Retrieving the retrieval method (dense or sparse) from the first argument
retrieval_method=$1

if [[ -z "$retrieval_method" != "sparse" && "$retrieval_method" != "dense" ]]; then
    echo "Error: Please provide a retrieval method (dense or sparse)."
    exit 1
fi

# Define datasets and k values
datasets=("FeTaQA" "QTSumm")
k_values=(1 3 5 10 20 50)

# Loop through datasets and k values
for dataset in "${datasets[@]}"; do
    for k in "${k_values[@]}"; do
        echo "Running: python main.py -d $dataset -rtr $retrieval_method -k $k"
        python main.py -d "$dataset" -rtr "$retrieval_method" -k "$k"
        if [ $? -ne 0 ]; then
            echo "Error: Command failed for dataset $dataset with k=$k"
            exit 1
        fi
    done
done

# Run visualization script
echo "Running: python visualization.py -rtr $retrieval_method"
python visualization.py -rtr "$retrieval_method"
if [ $? -ne 0 ]; then
    echo "Error: Visualization script failed"
    exit 1
fi

echo "All tasks completed successfully!"
