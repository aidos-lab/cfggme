"""Curvature analysis of strongly-regular graphs."""

import argparse
import logging
import warnings

import gudhi as gd
import gudhi.wasserstein
import networkx as nx
import numpy as np
import pandas as pd
from curvature import forman_curvature, ollivier_ricci_curvature, resistance_curvature
from python_log_indenter import IndentedLoggerAdapter
from scipy.stats import wasserstein_distance
from utils import (
    calculate_persistent_homology,
    propagate_edge_attribute_to_nodes,
    propagate_node_attribute_to_edges,
)


def degrees(G):
    """Calculate degrees vector."""
    return [deg for _, deg in nx.degree(G)]


def laplacian_eigenvalues(G):
    """Calculate Laplacian and return eigenvalues."""
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        return nx.laplacian_spectrum(G)


def pagerank(G):
    """Calculate page rank of nodes."""
    return [rank for _, rank in nx.pagerank(G).items()]


def prob_rw(G, node, node_to_index):
    """Probability measure based on random walk probabilities."""
    import scipy as sp

    A = nx.to_scipy_sparse_array(G, format="csr").todense()
    n, m = A.shape
    D = sp.sparse.csr_array(
        sp.sparse.spdiags(A.sum(axis=1), 0, m, n, format="csr")
    ).todense()

    P = np.linalg.inv(D) @ A

    values = np.zeros(len(G.nodes))
    values[node_to_index[node]] = 1.0

    x = values
    values = x + P @ x + P @ P @ x

    values /= values.sum()
    return values


def prob_two_hop(G, node, node_to_index):
    """Probability measure based on two-hop neighbourhoods."""
    alpha = 0.5
    values = np.zeros(len(G.nodes))
    values[node_to_index[node]] = alpha

    subgraph = nx.ego_graph(G, node, radius=2)

    w = 0.25

    direct_neighbors = list(G.neighbors(node))
    for neighbor in direct_neighbors:
        values[node_to_index[neighbor]] = (1 - alpha) * w

    w = 0.05

    for neighbor in subgraph.nodes():
        if neighbor not in direct_neighbors and neighbor != node:
            index = node_to_index[neighbor]
            values[index] = (1 - alpha) * w

    # TODO: Only necessary because I am making my life easy here.
    values /= values.sum()
    return values


def run_experiment(graphs, curvature_fn, prob_fn, k, node_level=False):
    """Run experiment on all graphs for a given probability function."""
    for graph in graphs:
        if prob_fn is not None:
            curvature = curvature_fn(graph, prob_fn=prob_fn)
        else:
            curvature = curvature_fn(graph)

        # Assign node-level attribute
        if node_level:
            curvature = {v: c for v, c in zip(graph.nodes(), curvature)}
            nx.set_node_attributes(graph, curvature, "curvature")

        # Assign edge-based attribute. This is the normal assignment
        # procedure whenever we are dealing with proper curvature
        # measurements.
        else:
            curvature = {e: c for e, c in zip(graph.edges(), curvature)}
            nx.set_edge_attributes(graph, curvature, "curvature")

    n_pairs = 0
    all_pairs = len(graphs) * (len(graphs) - 1) / 2

    for i, Gi in enumerate(graphs):
        for j, Gj in enumerate(graphs):
            if i < j:
                access_fn = (
                    nx.get_node_attributes if node_level else nx.get_edge_attributes
                )

                ci = list(access_fn(Gi, "curvature").values())
                cj = list(access_fn(Gj, "curvature").values())

                n_pairs += wasserstein_distance(ci, cj) > 1e-8

    log.add()
    log.info(f"Distinguishing {n_pairs}/{int(all_pairs)} pairs (raw)")
    log.sub()

    success_rate_raw = n_pairs / all_pairs

    persistence_diagrams = []

    for graph in graphs:
        if node_level:
            propagate_node_attribute_to_edges(graph, "curvature")
        else:
            propagate_edge_attribute_to_nodes(
                graph, "curvature", pooling_fn=lambda x: -1
            )

        diagrams = calculate_persistent_homology(graph, k=k)
        persistence_diagrams.append(diagrams)

    n_pairs = 0

    for i, Gi in enumerate(graphs):
        for j, Gj in enumerate(graphs):
            if i < j:
                distance = 0.0
                for D1, D2 in zip(persistence_diagrams[i], persistence_diagrams[j]):
                    distance += gudhi.bottleneck.bottleneck_distance(
                        np.asarray(D1), np.asarray(D2), e=1e-10
                    )

                n_pairs += distance > 1e-8

    log.add()
    log.info(f"Distinguishing {n_pairs}/{int(all_pairs)} pairs (TDA)")
    log.sub()

    success_rate_tda = n_pairs / all_pairs
    return success_rate_raw, success_rate_tda


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("FILE", type=str, help="Input file (in `.g6` format)")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="If set, store output in specified file.",
    )

    parser.add_argument(
        "-k",
        type=int,
        default=2,
        help="Specifies maximum expansion dimension for graphs.",
    )

    args = parser.parse_args()
    graphs = nx.read_graph6(args.FILE)

    prob_fns = [
        ("default", None),
        ("random_walk", prob_rw),
        ("two_hop", prob_two_hop),
    ]

    # Will collect rows for the output of the experimental table later
    # on. This makes it possible to "fire and forget" some jobs on the
    # cluster.
    rows = []

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    log = IndentedLoggerAdapter(logging.getLogger(__name__))

    log.info(f"Running experiment with {len(graphs)} graphs")

    log.info("Laplacian spectrum")
    log.add()

    e1, e2 = run_experiment(
        graphs, laplacian_eigenvalues, None, args.k, node_level=True
    )

    rows.append(
        {
            "name": "Laplacian spectrum",
            "raw": [e1],
            "tda": [e2],
        },
    )

    log.sub()

    log.info("Pagerank")
    log.add()

    e1, e2 = run_experiment(graphs, pagerank, None, args.k, node_level=True)

    rows.append(
        {
            "name": "Pagerank",
            "raw": [e1],
            "tda": [e2],
        },
    )

    log.sub()

    log.info("Degrees")
    log.add()

    e1, e2 = run_experiment(graphs, degrees, None, args.k, node_level=True)

    rows.append(
        {
            "name": "Degrees",
            "raw": [e1],
            "tda": [e2],
        },
    )

    log.sub()

    log.info("Forman--Ricci curvature")
    log.add()

    e1, e2 = run_experiment(graphs, forman_curvature, None, args.k)

    rows.append(
        {
            "name": "Forman--Ricci curvature",
            "raw": [e1],
            "tda": [e2],
        },
    )

    log.sub()
    log.info("Ollivier--Ricci curvature")

    log.add()

    for name, prob_fn in prob_fns:
        log.add()
        log.info(f"Probability measure: {name}")

        e1, e2 = run_experiment(graphs, ollivier_ricci_curvature, prob_fn, args.k)

        rows.append(
            {
                "name": "Ollivier--Ricci curvature",
                "prob": name,
                "raw": [e1],
                "tda": [e2],
            },
        )

        log.sub()

    log.sub()

    log.info("Resistance curvature")
    log.add()

    e1, e2 = run_experiment(graphs, resistance_curvature, None, args.k)

    rows.append({"name": "Resistance curvature", "raw": [e1], "tda": [e2]})

    log.sub()

    rows = [pd.DataFrame.from_dict(row) for row in rows]
    df = pd.concat(rows, ignore_index=True)
    print(df)

    if args.output is not None:
        df.to_csv(args.output, index=False)
