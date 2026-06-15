# ---------------------------------------------------
# Mobility LDP Analysis Toolkit
# mobility.py
#
# Author: Jay M. Appleton
# License: Apache-2.0
# ---------------------------------------------------

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

def compute_occupancy(df):
    """
    Compute spatio-temporal occupancy counts.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        - veh_id
        - t_bin
        - hex_id

    Returns
    -------
    pd.DataFrame
        Columns: hex_id, t_bin, count
    """

    occ = (
        df.groupby(["hex_id", "t_bin"])
          .size()
          .reset_index(name="count")
    )

    return occ

def compute_transitions(df):
    """
    Compute empirical transition counts from trajectory data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - veh_id
        - t_bin
        - hex_id

    Returns
    -------
    pd.DataFrame
        Columns:
        - hex_from
        - hex_to
        - count
    """

    # 1. enforce temporal ordering (critical for correctness)
    df_sorted = df.sort_values(["veh_id", "t_bin"])

    # 2. shift within each trajectory
    df_sorted["hex_from"] = df_sorted.groupby("veh_id")["hex_id"].shift(1)

    # 3. build transitions
    transitions = (
        df_sorted.dropna(subset=["hex_from"])
        .groupby(["hex_from", "hex_id"])
        .size()
        .reset_index(name="count")
        .rename(columns={"hex_id": "hex_to"})
    )

    return transitions

def compute_markov_kernel(transitions):
    """
    Convert transition counts into a Markov transition kernel.

    Parameters
    ----------
    transitions : pd.DataFrame
        Must contain:
        - hex_from
        - hex_to
        - count

    Returns
    -------
    pd.DataFrame
        Columns:
        - hex_from
        - hex_to
        - prob
    """

    # 1. compute total outgoing mass per origin state
    totals = (
        transitions.groupby("hex_from")["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "total"})
    )

    # 2. join totals back
    merged = transitions.merge(totals, on="hex_from", how="left")

    # 3. normalize
    merged["prob"] = merged["count"] / merged["total"]

    return merged[["hex_from", "hex_to", "prob"]]

def get_topk_hotspots(occupancy, k=10):
    """
    Compute top-k most frequently visited hex cells.

    Parameters
    ----------
    occupancy : pd.DataFrame
        Must contain:
        - hex_id
        - t_bin
        - count

    k : int
        number of top cells to return

    Returns
    -------
    pd.DataFrame
        Columns:
        - hex_id
        - total_count
    """

    # aggregate over time
    totals = (
        occupancy.groupby("hex_id")["count"]
        .sum()
        .reset_index(name="total_count")
    )

    # rank and select top-k
    topk = totals.sort_values("total_count", ascending=False).head(k)

    return topk

def compute_entropy(occupancy, normalize=True):
    """
    Compute spatial entropy over occupancy distributions.

    Parameters
    ----------
    occupancy : pd.DataFrame
        Must contain:
        - hex_id
        - t_bin
        - count

    normalize : bool
        If True, returns entropy per time bin normalized by log(N)

    Returns
    -------
    pd.DataFrame
        Columns:
        - t_bin
        - entropy
    """

    # total mass per time step
    totals = occupancy.groupby("t_bin")["count"].sum().reset_index(name="total")

    df = occupancy.merge(totals, on="t_bin", how="left")

    # probability
    df["p"] = df["count"] / df["total"]

    # entropy contribution
    df["plogp"] = df["p"] * np.log(df["p"])

    entropy = (
        df.groupby("t_bin")["plogp"]
        .sum()
        .reset_index(name="entropy")
    )

    entropy["entropy"] = -entropy["entropy"]

    # optional normalization (max entropy = log(N_t))
    if normalize:
        n_states = occupancy.groupby("t_bin")["hex_id"].nunique().reset_index(name="n")
        entropy = entropy.merge(n_states, on="t_bin")
        entropy["entropy"] = entropy["entropy"] / np.log(entropy["n"])

    return entropy[["t_bin", "entropy"]]

def to_sparse_matrix(transitions):
    """
    Convert transition table into sparse matrix form.

    Parameters
    ----------
    transitions : pd.DataFrame
        Must contain:
        - hex_from
        - hex_to
        - count

    Returns
    -------
    sparse_matrix : scipy.sparse.coo_matrix
        Transition count matrix
    index_map : dict
        hex_id -> matrix index
    """

    # 1. build index over all states
    states = pd.Index(
        pd.concat([transitions["hex_from"], transitions["hex_to"]]).unique()
    )

    index_map = {h: i for i, h in enumerate(states)}

    # 2. map to integer indices
    row = transitions["hex_from"].map(index_map).values
    col = transitions["hex_to"].map(index_map).values
    data = transitions["count"].values

    # 3. build sparse matrix
    mat = coo_matrix(
        (data, (row, col)),
        shape=(len(states), len(states))
    )

    return mat, index_map