#!/bin/bash

# Step 1: Check if all required Python files exist
required_files=("dataset.py" "main.py" "retrieval.py" "visualization.py")

for file in "${required_files[@]}"
do
    if [ ! -f "$file" ]; then
        echo "Error: Required file '$file' is missing."
        exit 1
    fi
done

# Step 2: Install required Python packages
pip install datasets faiss-gpu faiss-cpu matplotlib nltk numpy torch tqdm transformers

# Step 3: Create directories for results, retrieval indices, and graphs
mkdir -p results retrieval_indices graphs

echo "Setup complete!"
