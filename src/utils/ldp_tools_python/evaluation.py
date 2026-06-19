# ---------------------------------------------------
# Mobility LDP Analysis Toolkit
# evaluation.py
#
# Author: Jay M. Appleton
# License: Apache-2.0
# ---------------------------------------------------

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import os
import json
from datetime import datetime
import duckdb

def compute_metrics(
    occupancy_true,
    occupancy_ldp,
    transitions_true,
    transitions_ldp,
    kernel_true=None,
    kernel_ldp=None,
    hotspots_true=None,
    hotspots_ldp=None,
    entropy_true=None,
    entropy_ldp=None
):
    """
    Compute LDP utility metrics comparing true vs noisy mobility statistics.
    """

    metrics = {}

    # - 1. Transition distortion -----------------------

    merged = transitions_true.merge(
        transitions_ldp,
        on=["hex_from", "hex_to"],
        how="outer",
        suffixes=("_true", "_ldp")
    ).fillna(0)

    x = merged["count_true"].values.astype(float)
    y = merged["count_ldp"].values.astype(float)

    # --- normalize to probability distributions -------
    x_prob = x / (np.sum(x) + EPS)
    y_prob = y / (np.sum(y) + EPS)

    # --- Total Variation Distance ---------------------
    metrics["transition_tv"] = 0.5 * np.sum(np.abs(x_prob - y_prob))

    # --- Frobenius (keep as structural metric) -------
    metrics["transition_frobenius"] = np.sqrt(np.sum((x_prob - y_prob) ** 2))

    # (optional legacy/raw signal, but now clearly labeled)
    metrics["transition_l1_raw"] = np.sum(np.abs(x - y))

    # - 2. Kernel distortion  -------------------------
    if kernel_true is not None and kernel_ldp is not None:
        kt = kernel_true.set_index(["hex_from", "hex_to"])["prob"]
        kl = kernel_ldp.set_index(["hex_from", "hex_to"])["prob"]

        aligned = pd.concat([kt, kl], axis=1).fillna(0)
        aligned.columns = ["true", "ldp"]

        metrics["kernel_l1"] = np.abs(aligned["true"] - aligned["ldp"]).sum()
        metrics["kernel_frobenius"] = np.sqrt(
            ((aligned["true"] - aligned["ldp"]) ** 2).sum()
        )

    # - 3. Hotspot stability --------------------------
    if hotspots_true is not None and hotspots_ldp is not None:

        set_true = set(hotspots_true["hex_id"])
        set_ldp = set(hotspots_ldp["hex_id"])

        intersection = len(set_true & set_ldp)
        union = len(set_true | set_ldp)

        metrics["hotspot_jaccard"] = intersection / union if union > 0 else 0
        metrics["hotspot_overlap"] = intersection / len(set_true) if len(set_true) > 0 else 0

    # - 4. Entropy distortion -------------------------
    if entropy_true is not None and entropy_ldp is not None:

        merged_e = entropy_true.merge(
            entropy_ldp,
            on="t_bin",
            suffixes=("_true", "_ldp")
        )

        metrics["entropy_gap_mean"] = np.mean(
            merged_e["entropy_ldp"] - merged_e["entropy_true"]
        )

        metrics["entropy_gap_abs_mean"] = np.mean(
            np.abs(merged_e["entropy_ldp"] - merged_e["entropy_true"])
        )

    return metrics

def compare_occupancy(occupancy_true, occupancy_ldp, eps=1e-12):
    """
    Compare true vs LDP occupancy distributions.

    Returns KL, L1, and Jensen-Shannon-style divergence proxies.
    """

    metrics = {}

    # - 1. align tables -------------------------------
    true = occupancy_true.copy()
    ldp = occupancy_ldp.copy()

    # total per time bin
    true_tot = true.groupby("t_bin")["count"].sum().reset_index(name="total_true")
    ldp_tot = ldp.groupby("t_bin")["count"].sum().reset_index(name="total_ldp")

    true = true.merge(true_tot, on="t_bin")
    ldp = ldp.merge(ldp_tot, on="t_bin")

    # probabilities
    true["p"] = true["count"] / true["total_true"]
    ldp["p"] = ldp["count"] / ldp["total_ldp"]

    # - 2. align distributions ------------------------
    merged = true.merge(
        ldp,
        on=["hex_id", "t_bin"],
        how="outer",
        suffixes=("_true", "_ldp")
    ).fillna(0)

    p = merged["p_true"].values + eps
    q = merged["p_ldp"].values + eps

    # - 3. L1 error -----------------------------------
    metrics["l1_error"] = np.sum(np.abs(p - q))

    # - 4. KL divergence (true || ldp) ----------------
    metrics["kl_divergence"] = np.sum(p * np.log(p / q))

    # - 5. Jensen-Shannon (symmetric stability proxy) -
    m = 0.5 * (p + q)
    metrics["js_divergence"] = 0.5 * np.sum(p * np.log(p / m)) + \
                                0.5 * np.sum(q * np.log(q / m))

    # - 6. per-time-bin KL (for plotting)--------------
    kl_per_t = []

    for t in merged["t_bin"].unique():
        sub = merged[merged["t_bin"] == t]

        p_t = sub["p_true"].values + eps
        q_t = sub["p_ldp"].values + eps

        kl_t = np.sum(p_t * np.log(p_t / q_t))
        kl_per_t.append((t, kl_t))

    metrics["kl_per_tbin"] = kl_per_t
    metrics["kl_mean"] = np.mean([x[1] for x in kl_per_t])

    return metrics

def compare_transitions(transitions_true, transitions_ldp, eps=1e-12):
    """
    Compare true vs LDP transition structure.
    """

    metrics = {}

    # - 1. Align edge sets-----------------------------
    merged = transitions_true.merge(
        transitions_ldp,
        on=["hex_from", "hex_to"],
        how="outer",
        suffixes=("_true", "_ldp")
    ).fillna(0)

    x = merged["count_true"].values + eps
    y = merged["count_ldp"].values + eps

    # --- normalize ---
    x_norm = x / np.sum(x)
    y_norm = y / np.sum(y)

    # --- Total Variation (primary metric) ---
    metrics["transition_tv"] = 0.5 * np.sum(np.abs(x_norm - y_norm))

    # --- Frobenius (normalized structural distortion) ---
    metrics["transition_frobenius"] = np.sqrt(np.sum((x_norm - y_norm) ** 2))

    # --- Relative error (optional, but now consistent) ---
    metrics["transition_relative_error"] = np.sum(np.abs(x_norm - y_norm))

    # - 5. Edge distribution JS-style divergence ------
    x_norm = x / np.sum(x)
    y_norm = y / np.sum(y)

    m = 0.5 * (x_norm + y_norm)

    metrics["js_divergence"] = (
        0.5 * np.sum(x_norm * np.log(x_norm / m)) +
        0.5 * np.sum(y_norm * np.log(y_norm / m))
    )

    # - 6. Edge sparsity shift  -----------------------
    metrics["edge_sparsity_true"] = np.sum(x > eps)
    metrics["edge_sparsity_ldp"] = np.sum(y > eps)

    return metrics

def compare_kernels(kernel_true, kernel_ldp, eps=1e-12):
    """
    Compare true vs LDP Markov kernels.
    """

    metrics = {}

    # - 1. Align kernels on (i, j) --------------------
    merged = kernel_true.merge(
        kernel_ldp,
        on=["hex_from", "hex_to"],
        how="outer",
        suffixes=("_true", "_ldp")
    ).fillna(0)

    p = merged["prob_true"].values + eps
    q = merged["prob_ldp"].values + eps

    # - 2. Global L1 error ----------------------------
    metrics["l1_error"] = np.sum(np.abs(p - q))

    # - 3. Frobenius norm (matrix distortion) ---------
    metrics["frobenius"] = np.sqrt(np.sum((p - q) ** 2))

    # - 4. JS divergence (distributional stability) ---
    p_norm = p / np.sum(p)
    q_norm = q / np.sum(q)
    m = 0.5 * (p_norm + q_norm)

    metrics["js_divergence"] = (
        0.5 * np.sum(p_norm * np.log(p_norm / m)) +
        0.5 * np.sum(q_norm * np.log(q_norm / m))
    )

    # - 5. Row-wise KL --------------------------------
    row_kl = []

    for i in merged["hex_from"].unique():
        sub = merged[merged["hex_from"] == i]

        p_i = sub["prob_true"].values + eps
        q_i = sub["prob_ldp"].values + eps

        p_i = p_i / np.sum(p_i)
        q_i = q_i / np.sum(q_i)

        kl_i = np.sum(p_i * np.log(p_i / q_i))
        row_kl.append(kl_i)

    metrics["rowwise_kl_mean"] = np.mean(row_kl)
    metrics["rowwise_kl_max"] = np.max(row_kl)

    # - 6. Max transition distortion ------------------
    metrics["max_row_error"] = np.max(np.abs(p - q))

    return metrics

def compare_hotspots(hotspots_true, hotspots_ldp, k=10):
    """
    Compare top-k hotspot stability under LDP.
    """

    metrics = {}

    # - 1. ensure top-k lists -------------------------
    true_topk = hotspots_true["hex_id"].head(k).tolist()
    ldp_topk = hotspots_ldp["hex_id"].head(k).tolist()

    set_true = set(true_topk)
    set_ldp = set(ldp_topk)

    # - 2. Jaccard similarity (set overlap) -----------
    intersection = len(set_true & set_ldp)
    union = len(set_true | set_ldp)

    metrics["jaccard"] = intersection / union if union > 0 else 0

    # - 3. Top-k overlap ------------------------------
    metrics["overlap_k"] = intersection / k

    # - 4. Top-1 stability (most important cell) ------
    metrics["top1_stability"] = int(true_topk[0] == ldp_topk[0])

    # - 5. Rank correlation (if counts available) -----
    if "count" in hotspots_true.columns and "count" in hotspots_ldp.columns:

        merged = hotspots_true.merge(
            hotspots_ldp,
            on="hex_id",
            how="outer",
            suffixes=("_true", "_ldp")
        ).fillna(0)

        # take rank correlation over shared support
        rho, _ = spearmanr(
            merged["count_true"],
            merged["count_ldp"]
        )

        metrics["rank_correlation"] = rho

    return metrics

def run_sweeps(
    df_states,
    neighbor_map,
    epsilons,
    seeds,
    apply_ldp,
    compute_occupancy,
    compute_transitions,
    compute_markov_kernel,
    compute_entropy,
    compute_metrics,
    compare_occupancy,
    compare_transitions,
    compare_kernels,
    compare_hotspots,
    get_topk_hotspots
):
    """
    Full LDP experiment sweep over epsilon values.
    Returns:
        - results_df: STRICT scalar-only metrics (CSV-safe)
        - artifacts: full fidelity experimental outputs
    """

    import numpy as np
    import pandas as pd

    # ----------------------------
    # helper: keep only scalars
    # ----------------------------
    def to_scalar(v):
        if isinstance(v, (np.generic,)):
            return v.item()
        if isinstance(v, (int, float, bool)):
            return v
        return None  # drop anything non-scalar

    def filter_scalars(d):
        return {k: to_scalar(v) for k, v in d.items() if to_scalar(v) is not None}

    results = []

    # - baseline -----------------
    occ_true = compute_occupancy(df_states)
    trans_true = compute_transitions(df_states)
    kernel_true = compute_markov_kernel(trans_true)
    ent_true = compute_entropy(occ_true)
    hot_true = get_topk_hotspots(occ_true, k=10)

    # - artifacts ----------------
    artifacts = {
        "true_states": df_states,
        "trans_true": trans_true,
        "occ_true": occ_true,
        "kernel_true": kernel_true,
        "runs": []
    }

    # - sweep --------------------
    for eps in epsilons:
        for seed in seeds:

            # 1. LDP sample
            df_ldp = apply_ldp(df_states, neighbor_map, eps, seed)

            # 2. recompute structures
            occ_ldp = compute_occupancy(df_ldp)
            trans_ldp = compute_transitions(df_ldp)
            kernel_ldp = compute_markov_kernel(trans_ldp)
            hot_ldp = get_topk_hotspots(occ_ldp, k=10)

            # 3. compute comparisons
            occ_metrics = compare_occupancy(occ_true, occ_ldp)
            trans_metrics = compare_transitions(trans_true, trans_ldp)
            kernel_metrics = compare_kernels(kernel_true, kernel_ldp)
            hot_metrics = compare_hotspots(hot_true, hot_ldp)

            # 4. STRICT scalar filtering (core fix)
            row = {
                "epsilon": eps,
                "seed": seed,
                **filter_scalars(occ_metrics),
                **filter_scalars(trans_metrics),
                **filter_scalars(kernel_metrics),
                **filter_scalars(hot_metrics),
            }

            results.append(row)

            # 5. full artifacts (unchanged, safe)
            artifacts["runs"].append({
                "epsilon": eps,
                "seed": seed,
                "ldp_states": df_ldp,
                "occ_ldp": occ_ldp,
                "trans_ldp": trans_ldp,
                "kernel_ldp": kernel_ldp,
                "hot_ldp": hot_ldp
            })

    return pd.DataFrame(results), artifacts

def save_results(
    results_df,
    seed_results_df=None,
    out_dir="results_ldp_experiment",
    filename_prefix="ldp_mobility"
):
    """
    Save LDP experiment results in analysis-ready CSV format.

    This version ENFORCES scalar-only CSV outputs.
    Any non-scalar values are removed or converted.
    """

    import os
    import json
    import numpy as np
    import pandas as pd
    from datetime import datetime

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # - scalar safety ---------------------------------
    def to_scalar(v):
        if isinstance(v, (np.generic,)):
            return v.item()
        if isinstance(v, (int, float, bool, str)):
            return v
        return None  # drop non-scalars

    def sanitize_df(df):
        df = df.copy()
        for col in df.columns:
            df[col] = df[col].apply(to_scalar)
        return df

    # - 0. SANITIZE INPUTS ----------------------------
    results_df_clean = sanitize_df(results_df)

    seed_results_df_clean = (
        sanitize_df(seed_results_df) if seed_results_df is not None else None
    )

    # - 1. EPSILON-LEVEL SUMMARY ----------------------
    summary_path = os.path.join(
        out_dir,
        f"{filename_prefix}_summary_{timestamp}.csv"
    )
    results_df_clean.to_csv(summary_path, index=False)

    # - 2. SEED-LEVEL RESULTS -------------------------
    seed_path = None
    if seed_results_df_clean is not None:
        seed_path = os.path.join(
            out_dir,
            f"{filename_prefix}_seed_level_{timestamp}.csv"
        )
        seed_results_df_clean.to_csv(seed_path, index=False)

    # - 3. METADATA -----------------------------------
    metadata = {
        "timestamp": timestamp,
        "summary_rows": len(results_df_clean),
        "seed_rows": len(seed_results_df_clean) if seed_results_df_clean is not None else 0,
        "epsilons": sorted(results_df_clean["epsilon"].dropna().unique().tolist()),
        "has_seed_level": seed_results_df_clean is not None
    }

    meta_path = os.path.join(
        out_dir,
        f"{filename_prefix}_meta_{timestamp}.json"
    )

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # - 4. RETURN PATHS--------------------------------
    return {
        "summary_csv": summary_path,
        "seed_csv": seed_path,
        "meta_json": meta_path
    }