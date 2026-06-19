# ---------------------------------------------------
# Mobility LDP Analysis Toolkit
# ldp_core.py
#
# Author: Jay M. Appleton
# License: Apache-2.0
# ---------------------------------------------------

import h3
from h3 import grid_disk
import random
import math

def build_neighbor_map(hex_ids, k=1):
    return {
        h: list(grid_disk(h, k))
        for h in hex_ids
    }

def ldp_randomized_response(hex_id, neighbor_map, epsilon, rng):
    neighbors = neighbor_map[hex_id]
    k = len(neighbors)

    p = math.exp(epsilon) / (math.exp(epsilon) + k - 1)

    if rng.random() < p:
        return hex_id
    else:
        return random.choice([n for n in neighbors if n != hex_id])

def apply_ldp(df, neighbor_map, epsilon, seed):
    rng = random.Random(seed)

    df = df.copy()
    df["hex_id"] = df["hex_id"].apply(
        lambda x: ldp_randomized_response(x, neighbor_map, epsilon, rng)
    )
    return df        
