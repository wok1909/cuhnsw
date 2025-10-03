#!/bin/bash

# Usage: python make_dataset.py <dataset> <num query> <topk> <graph level> <ef_size> <GPUID>

RUN_PATH=/home/wok1909/workplace/RAG/hnsw/cuhnsw/examples/make_dataset.py
DATASETS=("siftsmall-128-euclidean")
QUERYS=(8)
TOPK=5
GRAPH_LEVEL=3
EF_SIZE=300
GPUID=3


for dataset in "${DATASETS[@]}"; do
    for query in "${QUERYS[@]}"; do
        echo "Running ${RUN_PATH} dataset=${dataset} query=${query} topk=${TOPK} graph_level=${GRAPH_LEVEL} ef_size=${EF_SIZE} gpu_id=${GPUID}"
        python ${RUN_PATH} ${dataset} ${query} ${TOPK} ${GRAPH_LEVEL} ${EF_SIZE} ${GPUID}
    done
done
