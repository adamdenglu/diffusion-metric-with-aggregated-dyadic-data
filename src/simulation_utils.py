# -*- coding: utf-8 -*-
# Copyright 2024 Tencent Inc.  All rights reserved.
# Author: adamdeng@tencent.com
import numpy as np
import networkx as nx
from functools import reduce
import networkx.algorithms.community as nx_comm


def total_treatment_effect(graph):
    '''
    Calculate the total treatment effect given graph and potential outcomes.

    Parameters
    ----------
    graph : networkx object
        networkx object representing the graph

    Returns
    ----------
    tau : float
        total treatment effect
    '''
    tau = 0
    for edge in graph.edges:
        tau += (
            graph.edges[edge]['eff_i'] +
            graph.edges[edge]['eff_j'] +
            graph.edges[edge]['eff_ij']
        )

    num_nodes = graph.number_of_nodes()
    tau /= num_nodes

    return tau


def network_data_generator(graph):
    '''
    Generate data directed from the adjacency matrix.

    Parameters
    ----------
    graph : networkx object
        networkx object representing the graph
    '''
    # generate the potential outcomes
    nx.set_node_attributes(graph, 0, "y_in_obs")
    nx.set_node_attributes(graph, 0, "y_out_obs")
    for edge in graph.edges:
        treat_i = graph.nodes[edge[0]]["z"]
        treat_j = graph.nodes[edge[1]]["z"]
        graph.edges[edge]['edge_obs'] = (
            graph.edges[edge]['eff_0'] +
            graph.edges[edge]['eff_i'] * treat_i +
            graph.edges[edge]['eff_j'] * treat_j +
            graph.edges[edge]['eff_ij'] * treat_i * treat_j
        )
        graph.nodes[edge[1]]["y_in_obs"] += graph.edges[edge]['edge_obs']
        graph.nodes[edge[0]]["y_out_obs"] += graph.edges[edge]['edge_obs']


def estimate(graph, prob, ratio):
    '''
    Calculte the estimators given graph and probability of treatment.

    Parameters
    ----------
    graph : networkx object
        networkx object representing the graph
    prob : float
        probability of treatment, between 0 and 1
    ratio : float
        ratio of experiment units to total population

    Returns
    ----------
    est_in : float
        estimator using incoming edges
    est_out : float
        estimator using outcoming edges
    est : float
        estimator for tte

    '''
    num_nodes = graph.number_of_nodes()
    # calculate estimators
    est_in = est_out = 0
    for node in graph.nodes:
        if graph.nodes[node]["in_expt"] == 0:
            continue
        if graph.nodes[node]["z"] == 1:
            est_in += graph.nodes[node]["y_in_obs"] / prob
            est_out += graph.nodes[node]["y_out_obs"] / prob
        else:
            est_in -= graph.nodes[node]["y_in_obs"] / (1 - prob)
            est_out -= graph.nodes[node]["y_out_obs"] / (1 - prob)
    est_in /= (num_nodes * ratio)
    est_out /= (num_nodes * ratio)
    est = est_in + est_out

    return est_in, est_out, est


def simulation(
    graph, reps, prob=0.5, ratio=1, model='lognormal', cluster=False
):
    '''
    Returns the estimators for simulated cases for 'reps' times with a fixed
    graph once generated

    Parameters
    ----------
    graph : str/networkx
        a str representing the type of graph or a networkx object
    reps : int
        simulation for 'reps' time
    prob : float
        probability of treatment, between 0 and 1
    ratio : float
        experiment ratio, between 0 and 1
    model : str
        set the generating distribution of the outcomes
    cluster : bool
        whether to use the clustering result to partition the traffic to expt
    '''
    est_in, est_out, est = (np.zeros(reps) for _ in range(3))

    # adj_mat = nx.to_numpy_array(graph)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    seed = 1
    if model == 'uniform':  # mimic the positive continous data
        # Potential effect of the i-th unit
        eff_i = np.random.default_rng(seed).uniform(0, 1, num_edges)
        # Potential effect of the j-th unit
        eff_j = np.random.default_rng(seed).uniform(0, 0.5, num_edges)
        # Potential interactive effect of the i-th and j-th units
        eff_ij = np.random.default_rng(seed).uniform(0, 0.5, num_edges)
        # Baseline without any treatment
        eff_0 = np.ones(shape=num_edges)

    elif model == 'binomial':  # mimic the count data
        # Potential effect of the i-th unit
        eff_i = np.random.default_rng(seed).binomial(1, 0.5, num_edges)
        # Potential effect of the i-th unit
        eff_j = np.random.default_rng(seed).binomial(1, 0.25, num_edges)
        # Potential interactive effect of the i-th and j-th units
        eff_ij = np.random.default_rng(seed).binomial(1, 0.25, num_edges)
        # Baseline without any treatment
        eff_0 = np.zeros(shape=num_edges)

    for k, edge in enumerate(graph.edges):
        graph.edges[edge]['eff_i'] = eff_i[k]
        graph.edges[edge]['eff_j'] = eff_j[k]
        graph.edges[edge]['eff_ij'] = eff_ij[k]
        graph.edges[edge]['eff_0'] = eff_0[k]

    tau = total_treatment_effect(graph)

    # clustering, fix the outcome of clustering
    if cluster:
        clusters = nx_comm.louvain_communities(graph, seed=10, resolution=10)
        clusters = sorted(clusters, key=len, reverse=True)
        num_cluster = len(clusters)

    for i in range(reps):
        if cluster:
            # assign the treatment to clusters
            in_expt_cluster = []
            while len(in_expt_cluster) == 0:
                cluster_rn = np.random.uniform(
                    low=0.0, high=1.0, size=num_cluster
                )
                in_expt_cluster = np.where(cluster_rn < ratio)[0]
            in_expt_nodes = reduce(
                lambda x, y: x.union(y), [clusters[i] for i in in_expt_cluster]
            )
            nx.set_node_attributes(graph, 0, "in_expt")
            nx.set_node_attributes(
                graph, {node: 1 for node in in_expt_nodes}, "in_expt"
            )

            rn = np.random.uniform(low=0.0, high=1.0, size=len(in_expt_nodes))
            tr_nodes = np.array(list(in_expt_nodes))[np.where(rn < prob)[0]]
            # tr_nodes = reduce(
            #     lambda x, y: x.union(y), [clusters[i] for i in tr_clusters]
            # )
            nx.set_node_attributes(graph, 0, "z")
            nx.set_node_attributes(graph, {node: 1 for node in tr_nodes}, "z")
        else:
            # assign the treatment to nodes
            rn = np.random.uniform(low=0.0, high=1.0, size=num_nodes)
            for k, node in enumerate(graph.nodes):
                graph.nodes[node]["in_expt"] = 1 * (rn[k] < ratio)
                graph.nodes[node]["z"] = 1 * (rn[k] < ratio * prob)

        network_data_generator(graph)
        results = estimate(graph, prob, ratio)
        est_in[i] = results[0]
        est_out[i] = results[1]
        est[i] = results[2]

    return est_in, est_out, est, tau
