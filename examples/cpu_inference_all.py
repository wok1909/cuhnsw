# Copyright (c) 2020 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,logging-format-truncated
import os
import sys
from os.path import join as pjoin
import time
import subprocess

import h5py
# import tqdm
import numpy as np
import pandas as pd

import hnswlib

NUM_DATA = 1000000
DATA_FILE = "sift-128-euclidean.hdf5"
DIST_TYPE = "l2"

BARRIER_SIZE = 100
RES_DIR = "/home/ubuntu/data/sift-128-euclidean"
INDEX_FILE = "cuhnsw.index"
CUHNSW_INDEX_FILE = "cuhnsw.index"
HNSWLIB_INDEX_FILE = "hnswlib.index"
DATA_URL = f"http://ann-benchmarks.com/{DATA_FILE}"
NRZ = DIST_TYPE == "cosine"
OPT = { \
    "c_log_level": 2,
    "ef_construction": 150,
    "hyper_threads": 100,
    "block_dim": 128,
    "nrz": NRZ,
    "reverse_cand": False,
    "heuristic_coef": 0.0,
    "dist_type": DIST_TYPE, \
}

def run_cpu_inference(topk=5, ef_search=300, index_file=INDEX_FILE,
                      evaluate=True, num_threads=8, warmup=0):
  print("=" * BARRIER_SIZE)
  print("(ns)")
  data_path = pjoin(RES_DIR, DATA_FILE)
  
#   print(f"cpu inference on {data_path} with index {index_path}")
  h5f = h5py.File(data_path, "r")
  num_data = h5f["train"].shape[0]
  queries = h5f["test"][:, :].astype(np.float32)
  neighbors = h5f["neighbors"][:, :topk].astype(np.int32)
  h5f.close()
  hl0 = hnswlib.Index(space=DIST_TYPE, dim=queries.shape[1])
#   print(f"load {index_path} by hnswlib")
  num_queries = queries.shape[0]
  index_path = pjoin(RES_DIR, index_file)
  print(f"Index path: {index_path}")
  print(f"Num query: {num_queries}")
  hl0.load_index(index_path, max_elements=num_data)
  hl0.set_ef(ef_search)
  if NRZ:
    queries /= np.linalg.norm(queries, axis=1)[:, None]

  result_times=[]
  for i in range(warmup):
    labels, _ = hl0.knn_query(queries, k=topk, num_threads=num_threads)
  hl0.stat_cycle_clear()
  start = time.time()
  labels, _ = hl0.knn_query(queries, k=topk, num_threads=num_threads)
  el0 = time.time() - start
  result_times.append(el0)
  hl0.print_elapsed_time(1)
  # print(f"Avg time: {sum(result_times)/len(result_times) * 1e9:.4f}")
  print(f"Time: {el0}")
  
  return el0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <num_threads> <warmup>")
        sys.exit(1)

    num_threads = int(sys.argv[1])
    warmup = int(sys.argv[2])

    print(f"Query: ALL, nThreads: {num_threads}, warmup: {warmup}")
    run_cpu_inference(num_threads=num_threads, warmup=warmup)