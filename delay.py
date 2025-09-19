"""Compute inter-satellite routing delay from pre-generated HDF5 data."""

from __future__ import annotations

import h5py
import networkx as nx
import numpy as np


def delay(constellation_name, source, target, shell_name, t):
    """Return round-trip delay (ms) between two satellites for slot ``t``."""

    file_path = f"data/XML_constellation/{constellation_name}.h5"
    with h5py.File(file_path, "r") as file:
        dataset = file["delay"][shell_name][f"timeslot{t}"]
        matrix = np.asarray(dataset, dtype=float)

    graph = nx.Graph()
    node_count = matrix.shape[0] - 1
    graph.add_nodes_from(f"satellite_{i}" for i in range(1, node_count + 1))

    for i in range(1, node_count + 1):
        for j in range(i + 1, node_count + 1):
            weight = matrix[i, j]
            if weight > 0:
                graph.add_edge(f"satellite_{i}", f"satellite_{j}", weight=weight)

    start = f"satellite_{source.id}"
    end = f"satellite_{target.id}"
    return 2 * nx.dijkstra_path_length(graph, source=start, target=end)
