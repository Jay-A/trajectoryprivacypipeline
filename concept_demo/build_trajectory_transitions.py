import argparse
import duckdb

import argparse

def parse_args():
    USAGE = """
    Example usage:

    python3 build_trajectory_transitions.py -i statecount.duckdb -r 9 -o trajectory_transitions \
        --drop-existing

    Optional flags:
        --max-time-gap 1
        --verbose
        --dry-run
    """

    parser = argparse.ArgumentParser(
        description="Build trajectory transition tables from state sequences"
    )

    # - Core inputs ----------------------------------------------
    parser.add_argument("-i", "--input", required=True,
                        help="Path to DuckDB database (statecount.duckdb)")

    parser.add_argument("-r", "--res", type=int, default=9,
                        help="H3 resolution to process (must match existing trajectory_states_r{r})")

    # - Output control -------------------------------------------
    parser.add_argument("-o", "--output", default="trajectory_transitions.duckdb",
                        help="Path to DuckDB database out")

    # - Rebuild / cleanup control --------------------------------
    parser.add_argument("--drop-existing", action="store_true",
                        help="Drop existing transition table before building")

    parser.add_argument("--reset", action="store_true",
                        help="Alias for --drop-existing")

    # - Temporal correctness controls ----------------------------
    parser.add_argument("--max-time-gap", type=int, default=1,
                        help="Maximum allowed t_bin gap between consecutive states (prevents fake jumps)")

    # - Debug / observability ------------------------------------
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose logging")

    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be executed without writing to DB")

    parser.add_argument(
        "--table-name", default=None, 
        help="Output table name inside the DuckDB database. "
             "If not provided, defaults to transitions_dt{max_gap}_r{res}"
    )

    return parser.parse_args()

def validate_time_continuity(con, table, verbose=True):
    # ------------------------------------------------------------
    # 1. DUPLICATE CHECK (must fail)
    # ------------------------------------------------------------
    dup_keys = con.execute(f"""
        SELECT veh_id, t_bin
        FROM {table}
        GROUP BY veh_id, t_bin
        HAVING COUNT(*) > 1
    """).fetchall()

    # ------------------------------------------------------------
    # 2. DEBUG SAMPLE (optional)
    # ------------------------------------------------------------
    dup_sample = con.execute(f"""
        SELECT *
        FROM {table}
        WHERE (veh_id, t_bin) IN (
            SELECT veh_id, t_bin
            FROM {table}
            GROUP BY veh_id, t_bin
            HAVING COUNT(*) > 1
        )
        ORDER BY veh_id, t_bin
        LIMIT 50
    """).fetchdf()

    print(dup_sample)

    # ------------------------------------------------------------
    # 3. FAIL CONDITION
    # ------------------------------------------------------------
    if len(dup_keys) > 0:
        raise ValueError(
            f"Duplicate (veh_id, t_bin) entries found: {len(dup_keys)}"
        )

    # ------------------------------------------------------------
    # 2. ORDERING CHECK (must fail)
    # ------------------------------------------------------------
    bad_order = con.execute(f"""
        WITH diffs AS (
            SELECT
                veh_id,
                t_bin - LAG(t_bin) OVER (
                    PARTITION BY veh_id
                    ORDER BY t_bin
                ) AS dt
            FROM {table}
        )
        SELECT DISTINCT veh_id
        FROM diffs
        WHERE dt < 0
    """).fetchall()

    if bad_order:
        raise ValueError(
            f"Time ordering violation in {len(bad_order)} vehicles"
        )

    # ------------------------------------------------------------
    # 3. COVERAGE STATS (diagnostic only)
    # ------------------------------------------------------------
    if verbose:
        stats = con.execute(f"""
            SELECT
                veh_id,
                MIN(t_bin) AS t_min,
                MAX(t_bin) AS t_max,
                COUNT(*) AS n_obs,
                (MAX(t_bin) - MIN(t_bin) + 1) AS span,
                (MAX(t_bin) - MIN(t_bin) + 1 - COUNT(*)) AS missing_bins
            FROM {table}
            GROUP BY veh_id
            ORDER BY missing_bins DESC
            LIMIT 20
        """).fetchdf()

        print("\n[TRAJECTORY COVERAGE SUMMARY]")
        print(stats)

    if verbose:
        print("[VALIDATION OK] Markov structure is valid (duplicates removed, ordering correct)")

    return True

# def build_transitions(con, state_table, transition_table, args):
#     max_gap = args.max_time_gap

#     query = f"""
#     CREATE TABLE {transition_table} AS
#     WITH ordered AS (
#         SELECT
#             veh_id,
#             t_bin,
#             hex_id,
#             LAG(hex_id) OVER (
#                 PARTITION BY veh_id
#                 ORDER BY t_bin
#             ) AS hex_from,
#             LAG(t_bin) OVER (
#                 PARTITION BY veh_id
#                 ORDER BY t_bin
#             ) AS prev_t_bin
#         FROM input_db.trajectory_states_r{args.res}
#     )
#     SELECT
#         veh_id,
#         t_bin AS t_bin,
#         hex_from,
#         hex_id AS hex_to
#     FROM ordered
#     WHERE
#         hex_from IS NOT NULL
#         AND (t_bin - prev_t_bin) <= {max_gap}
#     """

#     if args.dry_run:
#         print(query)
#         return

#     con.execute(query)   

def build_transitions(con, state_table, transition_table, args):
    max_gap = args.max_time_gap

    query = f"""
    CREATE TABLE {transition_table} AS
    WITH ordered AS (
        SELECT
            veh_id,
            t_bin,
            hex_id,
            LAG(hex_id) OVER (
                PARTITION BY veh_id
                ORDER BY t_bin
            ) AS hex_from,
            LAG(t_bin) OVER (
                PARTITION BY veh_id
                ORDER BY t_bin
            ) AS prev_t_bin
        FROM {state_table}
    )
    SELECT
        veh_id,
        t_bin,
        hex_from,
        hex_id AS hex_to
    FROM ordered
    WHERE
        hex_from IS NOT NULL
        AND (t_bin - prev_t_bin) <= {max_gap}
    """

    if args.dry_run:
        print(query)
        return

    con.execute(query)     

def run_pipeline(args):
    con = duckdb.connect(args.input)

    state_table = f"trajectory_states_r{args.res}"

    transition_table = (
        args.table_name
        or f"trajectory_transitions_r{args.res}"
    )

    validate_time_continuity(con, state_table, args.verbose)

    if args.drop_existing:
        con.execute(f"DROP TABLE IF EXISTS {transition_table}")

    build_transitions(con, state_table, transition_table, args)

# def run_pipeline(args):
#     con = duckdb.connect(args.output)
#     con.execute(f"ATTACH '{args.input}' AS input_db")
#     state_table = f"input_db.trajectory_states_r{args.res}"

#     transition_table = (
#         args.table_name
#         or f"transitions_gap{args.max_time_gap}_r{args.res}"
#     )

#     validate_time_continuity( con, state_table, args.verbose )

#     if args.drop_existing:
#         drop_transition_table(con, transition_table)

#     build_transitions(con, state_table, transition_table, args)

#     return;

def main():
    args = parse_args()
    run_pipeline(args)

if __name__ == "__main__":
	main()