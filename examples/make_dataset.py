# Copyright (c) 2020 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,logging-format-truncated
import os, sys
from os.path import join as pjoin
import json
import time
import subprocess

import h5py
import fire
# import tqdm
import numpy as np
import pandas as pd

import hnswlib
from cuhnsw import aux, CuHNSW

DATASETS = {"siftsmall-128-euclidean": "l2",
            "30_siftsmall-128-euclidean": "l2",
            "100_siftsmall-128-euclidean": "l2",
            "1000_siftsmall-128-euclidean": "l2",
            "3000_siftsmall-128-euclidean": "l2",
            "siftsmall-128-euclidean_extend_q2048": "l2",}
URL = {"siftsmall-128-euclidean": "https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/siftsmall-128-euclidean.hdf5",}

LOGGER = aux.get_logger()

BARRIER_SIZE = 100
# RES_DIR = "res"
# RES_DIR = "/home/wok1909/workplace/RAG/ndpxsim/data"
RES_DIR = "/workspace/data" 
INDEX_FILE = "hnswlib.index"
CUHNSW_INDEX_FILE = "cuhnsw.index"
HNSWLIB_INDEX_FILE = "hnswlib.index"



def download(url, dataset, base_path):
  DATA_DIR = pjoin(RES_DIR, dataset)
  if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)
  if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

  data_path = pjoin(DATA_DIR, dataset+".hdf5")
  if os.path.exists(data_path):
    return
  cmds = ["wget", url, "-O", data_path + ".tmp"]
  cmds = " ".join(cmds)
  LOGGER.info("download data: %s", cmds)
  subprocess.call(cmds, shell=True)
  os.rename(data_path + ".tmp", data_path)

def dump_config(opt, path):
  config_path = pjoin(path, "config.json")
  with open(config_path, "w") as f:
    json.dump(opt, f, indent=2)
  LOGGER.info("dump config to %s", config_path)
  return config_path

def run_gpu_training(opt, path):
  print("=" * BARRIER_SIZE)
  data_path = pjoin(RES_DIR, opt["dataset"], opt["dataset"] + ".hdf5")
  data_dir = path
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  LOGGER.info("gpu training on %s with ef const %d", data_path, opt["ef_construction"])
  
  run_opt = opt.copy()
  del run_opt["dataset"]
  del run_opt["num_query"]
  del run_opt["topk"]

  ch0 = CuHNSW(run_opt)
  h5f = h5py.File(data_path, "r")

  print(f"DATAPATH: {data_path}")
  data = h5f["train"][:, :].astype(np.float32)
  test_data = h5f["test"][:, :].astype(np.float32)
  num_query = opt["num_query"]
  max_test = 512
  if num_query <= max_test:
    queries = test_data[:num_query]
  else:
    queries = test_data[:]
    extra_needed = num_query - max_test
    repeat_queries = test_data[:extra_needed]
    queries = np.vstack([queries, repeat_queries])
  queries = queries.astype(np.float32)
  h5f.close()
  ch0.set_data(data)
  start = time.time()
  ch0.build()
  el0 = time.time() - start
  LOGGER.info("elpased time to build by cuhnsw: %.4e sec", el0)
  index_path = pjoin(data_dir, CUHNSW_INDEX_FILE)
  ch0.save_index(index_path)

  index_path = pjoin(data_dir, "text_"+CUHNSW_INDEX_FILE)
  ch0.save_index_as_text(index_path, queries, data)
  return el0

def run_gpu_inference(opt, path):
  print("=" * BARRIER_SIZE)
  # data_path = pjoin(RES_DIR, DATA_FILE)
  data_path = data_path = pjoin(RES_DIR, opt["dataset"], opt["dataset"] + ".hdf5")
  data_dir = path
  index_path = pjoin(data_dir, CUHNSW_INDEX_FILE)
  LOGGER.info("gpu inference on %s with index %s", data_path, index_path)
  
  run_opt = opt.copy()
  del run_opt["dataset"]
  del run_opt["num_query"]
  del run_opt["topk"]

  ch0 = CuHNSW(run_opt)
  LOGGER.info("load model from %s by cuhnsw", index_path)
  ch0.load_index(index_path)

  h5f = h5py.File(data_path, "r")
  data = h5f["train"][:, :].astype(np.float32)
  queries = h5f["test"][:opt['num_query'], :].astype(np.float32)
  h5f.close()
  if opt['nrz']:
    data /= np.linalg.norm(data, axis=1)[:, None]

  print("Queries: ", queries)

  nns, distances, found_cnt = ch0.search_knn(queries[:opt['num_query']], opt['topk'], opt['ef_construction'], data_dir)
  for idx, (nn0, distance, cnt) in \
      enumerate(zip(nns, distances, found_cnt)):
    print("=" * BARRIER_SIZE)
    print(f"query {idx + 1}")
    print("-" * BARRIER_SIZE)
    for _idx, (_nn, _dist) in enumerate(zip(nn0[:cnt], distance[:cnt])):
      if opt['dist_type'] == "l2":
        real_dist = np.linalg.norm(data[_nn] - queries[idx])
        _dist = np.sqrt(_dist)
      elif opt['dist_type'] == "dot":
        real_dist = data[_nn].dot(queries[idx])
      print(f"rank {_idx + 1}. neighbor: {_nn}, dist by lib: {_dist}, "
            f"actual dist: {real_dist}")

if __name__ == "__main__":

  args = sys.argv[1:]
  if len(args) != 6:
    print("Usage: python make_dataset.py <dataset> <num query> <topk> <graph level> <ef_size> <GPUID>")
    sys.exit(1)

  dataset = args[0]
  num_query = int(args[1])
  topk = int(args[2])
  graph_level = int(args[3])
  ef_size = int(args[4])
  gpu_id = int(args[5])

  os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

  if dataset not in DATASETS.keys():
    print(f"Dataset not supported. Please choose from: {DATASETS}")

  dist_type = DATASETS[dataset]
  # url = URL[dataset]

  nrz = dist_type == "cosine"
  opt = { \
    "dataset": dataset,
    "num_query": num_query,
    "topk": topk,
    "c_log_level": graph_level,
    "ef_construction": ef_size,
    "hyper_threads": 100,
    "block_dim": 32,
    "nrz": nrz,
    "reverse_cand": False,
    "heuristic_coef": 0.0,
    "dist_type": dist_type
  }

  dataset_path = pjoin(RES_DIR, dataset)
  experiment_path = pjoin(dataset_path, f"q{num_query}_top{topk}_l{graph_level}_ef{ef_size}")

  if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
  if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)

  # download(url, dataset, dataset_path)

  dump_config(opt, experiment_path)

  run_gpu_training(opt, experiment_path)

  run_gpu_inference(opt, experiment_path)