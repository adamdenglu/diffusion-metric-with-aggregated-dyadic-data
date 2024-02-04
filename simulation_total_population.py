# -*- coding: utf-8 -*-
# Copyright 2024 Tencent Inc.  All rights reserved.
# Author: adamdeng@tencent.com
import os
import numpy as np
import pandas as pd
import networkx as nx
from time import time
from joblib import Parallel, delayed
from src.simulation_utils import simulation


def run_simulation(
    prob=0.5, ratio=1, rep=100, model='lognormal', dataset="cornell"
):
    if dataset == "cornell":
        path = "./dataset/socfb-Cornell5.mtx"
    elif dataset == "stanford":
        path = "./dataset/socfb-Stanford3.mtx"
    df_graph = pd.read_table(
        path, skiprows=1, names=["source", "target"], sep=" "
    )
    graph = nx.from_pandas_edgelist(df_graph)

    est_in, est_out, est, tau = simulation(graph, rep, prob, ratio, model)
    # save results
    path = f"./results/all_population/{dataset}/"
    if not os.path.exists(path):
        os.makedirs(path)
    rnd = int(time() * 1e8 % 1e8)
    save_path = path + f"{model}_results_{rnd}.npz"
    np.savez(
        save_path,
        # estimation results
        est_in=est_in,
        est_out=est_out,
        est=est,
        # true te
        tau=tau,
        # set up
        prob=prob,
        ratio=ratio,
        model=model,
        dataset=dataset,
    )


if __name__ == "__main__":
    probs = np.linspace(0.1, 0.9, 5)
    ratio = 1
    rep = 100

    models = ["uniform", "binomial"]
    datasets = ["cornell", "stanford"]
    begin_time = time()
    Parallel(n_jobs=-1, verbose=5)(
        delayed(run_simulation)(prob, ratio, rep, model, dataset)
        for prob in probs for model in models for dataset in datasets
        for s in range(100)
    )
    print(f"Finished in {time() - begin_time} seconds.")
