#!/bin/bash

DATASET=""
SPLITS="train,validation,test"
RETRIEVAL=""
ELEMENTS="title,header,metadata,table,full"
K="1,3,5,10,20,50"

for ARG in "$@"
do
    case $ARG in
        DATASET=*) DATASET="${ARG#*=}" ;;
        SPLITS=*) SPLITS="${ARG#*=}" ;;
        RETRIEVAL=*) RETRIEVAL="${ARG#*=}" ;;
        ELEMENTS=*) ELEMENTS="${ARG#*=}" ;;
        K=*) K="${ARG#*=}" ;;
        *) echo "Error: $ARG is unknown paramter."; exit 1 ;;
    esac
done

if [[ -z $DATASET ]]; then
    echo "Error: dataset is required."
    exit 1
fi

if [[ -z $RETRIEVAL ]]; then
    echo "Error: retrieval method is required."
    exit 1
fi

python main.py -d "$DATASET" -sp "$SPLITS" -rtr "$RETRIEVAL" -elem "$ELEMENTS" -k "$K" || python3 main.py -d "$DATASET" -sp "$SPLITS" -rtr "$RETRIEVAL" -elem "$ELEMENTS" -k "$K"
