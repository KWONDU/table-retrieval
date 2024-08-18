#!/bin/bash

required_files=("dataset.py" "main.py" "retrieval.py" "util.py" "visualization.py")
for file in "${required_files[@]}"
do
    if [[ ! -f $file ]]; then
        echo "Error: Required file '$file' is missing."
        exit 1
    fi
done

pip install datasets faiss-gpu faiss-cpu matplotlib nltk numpy torch tqdm transformers
python -c "import nltk; nltk.download('stopwords')"

mkdir -p results retrieval_indices graphs

echo "Setup complete!"
