#!/bin/bash

# num_querys=(1 4 8 16 32 64 128 256 512 1024 2048)
# num_threads=(1 2 4 8 16 32 64)
# num_cores=(1 2 4 8 16 32)

# warmup=0

# for num_core in "${num_cores[@]}"; do
#     for num_query in "${num_querys[@]}"; do
#         for num_thread in "${num_threads[@]}"; do
#             echo "Running query ${num_query}, nthreads ${num_thread}"
            
#             core_range="0-$((num_core - 1))"

#             echo "taskset -c ${core_range} python3 /home/ubuntu/cuhnsw/examples/cpu_inference_warmup.py ${num_query} ${num_thread} ${warmup} > /home/ubuntu/results/${DATASET}/core${num_core}/q${num_query}_t${num_thread}.log"
#             taskset -c ${core_range} python3 /home/ubuntu/cuhnsw/examples/cpu_inference_warmup.py ${num_query} ${num_thread} ${warmup} #> /home/ubuntu/results/${DATASET}/core${num_core}/q${num_query}_t${num_thread}.log
#         done
#     done
# done

# # All
# for num_core in "${num_cores[@]}"; do
#     for num_thread in "${num_threads[@]}"; do
#         echo "Running query ${num_query}, nthreads ${num_thread}"
        
#         core_range="0-$((num_core - 1))"

#         echo "taskset -c ${core_range} python3 /home/ubuntu/cuhnsw/examples/cpu_inference_all.py ${num_thread} ${warmup} > /home/ubuntu/results/${DATASET}/core${num_core}/t${num_thread}.log"
#         taskset -c ${core_range} python3 /home/ubuntu/cuhnsw/examples/cpu_inference_all.py ${num_thread} ${warmup} #> /home/ubuntu/results/${DATASET}/core${num_core}/q${num_query}_t${num_thread}.log
#     done
# done


DATASET="coco-i2i-512-angular"


declare -A QUERY_WARMUP=(
    [1]=16
    [4]=16
    [8]=16
    [16]=16
    [32]=16
    [64]=16
    [128]=8
    [256]=8
    [512]=4
    [1024]=2
    [2048]=1
    [4096]=1
    [8192]=1
    [10000]=1
)
NUM_TRIES=30
num_queries=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
num_threads=(1 2 4 8 16 32 64)
num_cores=(1 2 4 8 16)

mkdir -p /home/ubuntu/results/${DATASET}
for num_core in "${num_cores[@]}"; do
    core_range="0-$((num_core - 1))"
    mkdir -p /home/ubuntu/results/${DATASET}/core${num_core}

    for num_query in "${num_queries[@]}"; do
        warmup=${QUERY_WARMUP[$num_query]}

        for num_thread in "${num_threads[@]}"; do
            echo "Running q=${num_query}, warmup=${warmup}, threads=${num_thread}, cores=${num_core}"
            rm /home/ubuntu/results/${DATASET}/core${num_core}/q${num_query}_t${num_thread}.log
            for try_idx in $(seq 1 ${NUM_TRIES}); do
                echo "taskset -c ${core_range} python3 /home/ubuntu/cuhnsw/examples/cpu_inference_warmup.py ${num_query} ${num_thread} ${warmup} >> /home/ubuntu/results/${DATASET}/core${num_core}/q${num_query}_t${num_thread}.log"
                taskset -c ${core_range} python3 /home/ubuntu/cuhnsw/examples/cpu_inference_warmup.py ${num_query} ${num_thread} ${warmup} >> /home/ubuntu/results/${DATASET}/core${num_core}/q${num_query}_t${num_thread}.log
            done
        done
    done
done