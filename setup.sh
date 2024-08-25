#!/bin/bash

required_files=("dataset.py" "main.py" "retrieval.py" "util.py" "visualization.py" "source_data/openwikitable.tar.gz")
for file in "${required_files[@]}"
do
    if [[ ! -f $file ]]; then
        echo "Error: Required file '$file' is missing."
        exit 1
    fi
done

pip install datasets faiss-gpu faiss-cpu matplotlib nltk numpy torch tqdm transformers
python -c "import nltk; nltk.download('stopwords'), download_dir='nltk_data'"

mkdir -p results retrieval_indices graphs
tar -xzvf source_data/openwikitable.tar.gz -C source_data/

echo "Setup complete!"
