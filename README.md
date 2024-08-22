# Table Retrieval

![naive_framework](markdown/navie_framework.png)

## 1. Setup

    ./setup.sh

## 2. Run

    ./run.sh DATASET={dataset name} \
             SPLITS={dataset splits, optional} \
             RETRIEVAL={retrieval method} \
             ELEMENTS={tabular data elements, optional} \
             K={top-k tables, optional}

### 2.1 dataset name

FeTaQA, QTSumm, OpenWikiTable

### 2.2 dataset splits

train, validation, test
> default: "train,validation,split"

### 2.3 retrieval method

bm25, dpr

### 2.4 tabular data elements

title, header, metadata, table, full
> default: "title,header,metadata,table,full"

### 2.5 top-k tables

any natural numbers
> default: 1,3,5,10,20,50

## 3. Directory

### 3.1 graphs

png

### 3.2 results

txt

### 3.3 retrieval_indices (only local)

csv
