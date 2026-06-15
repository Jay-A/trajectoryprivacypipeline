# ---------------------------------------------------
# Mobility LDP Analysis Toolkit
# build_state_space.py
#
# Author: Jay M. Appleton
# License: Apache-2.0
# ---------------------------------------------------

import argparse 
import duckdb
import pyproj
import h3
import pyarrow as pa

def parse_args():

    USAGE = """
    Example usage:

    python3 build_state_space.py -i berlin.test_h3.duckdb -o statecount.duckdb --res 9 --overwrite
    """

    parser = argparse.ArgumentParser(
        description="Build LDP-ready spatiotemporal state space from H3 trajectory data",
        epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter )

    parser.add_argument( "-i", "--input", required=True,
        help="Input raw DuckDB file  " )

    parser.add_argument( "-o", "--output", default="statecount.duckdb",
        help="Output DuckDB file for state space" )

    parser.add_argument( "-t", "--timestep", type=int, default=30,
        help="Time bin size in seconds (default: 30)" )

    parser.add_argument("-r", "--res", type=int, default=9,
        help="Target H3 resolution for STATE SPACE (not raw data, which is fixed at r9)")

    parser.add_argument( "--time-column", default="time",
        help="Column containing timestamp" )

    parser.add_argument(
        "--rebuild-raw", action="store_true",
        help="Force rebuild of r9 raw state table" )

    parser.add_argument( "--overwrite", action="store_true",
        help="Drop existing output tables before creating new ones" )

    parser.add_argument(
        "--agg", type=str, default="last", choices=["last", "max"],
        help="State collapse strategy within (veh_id, t_bin): 'last' (temporal) or 'max' (most frequent)"
    )

    return parser.parse_args()

def build_trajectory_states_r9(con, args, raw_table="input_db.fcd_segmented", target_table="trajectory_states_r9"):

    NET_X = 363050.47
    NET_Y = 5798536.79

    SOURCE_CRS = "EPSG:32633"
    TARGET_CRS = "EPSG:4326"
    H3_RES = 9

    print("[R9] building trajectory state space...")

    # --- READ RAW DATA (ONLY SOURCE OF TRUTH) ---
    df = con.execute(f"""
        SELECT veh_id, time, x, y, speed
        FROM {raw_table}
    """).fetchdf()

    # --- COORD SHIFT ---
    df["x"] += NET_X
    df["y"] += NET_Y

    # --- PROJECTION ---
    transformer = pyproj.Transformer.from_crs(
        SOURCE_CRS,
        TARGET_CRS,
        always_xy=True
    )

    lon, lat = transformer.transform(df["x"].values, df["y"].values)

    df["hex_id"] = [
        str(h3.latlng_to_cell(la, lo, H3_RES))
        for la, lo in zip(lat, lon)
    ]

    df["hex_id"] = df["hex_id"].astype("string")

    df["t_bin"] = (df["time"] // args.timestep).astype("int64")

    # --- AGGREGATION ---
    if args.agg == "last":
        df = df.sort_values("time")
        df = df.drop_duplicates(subset=["veh_id", "t_bin"], keep="last")

    elif args.agg == "max":
        df = (
            df.groupby(["veh_id", "t_bin", "hex_id"])
              .size()
              .reset_index(name="c")
              .sort_values("c", ascending=False)
              .drop_duplicates(subset=["veh_id", "t_bin"], keep="first")
        )
    else:
        raise ValueError(args.agg)

    df = df[["veh_id", "t_bin", "hex_id"]]

    # --- WRITE TO DUCKDB SAFELY ---
    con.execute(f"DROP TABLE IF EXISTS {target_table}")

    table = pa.Table.from_pandas(df, preserve_index=False)
    con.register("tmp_r9", table)

    con.execute(f"""
        CREATE TABLE {target_table} AS
        SELECT * FROM tmp_r9
    """)

    # --- WRITE METADATA (COUPLED ARTIFACT) ---
    con.execute(f"""
        CREATE OR REPLACE TABLE state_metadata_r9 AS
        SELECT
            9 AS h3_resolution,
            ? AS timestep_seconds,
            'hex_id' AS h3_column,
            'trajectory_states_r9' AS source_table
    """, [args.timestep])

    con.unregister("tmp_r9")

    print(f"[R9] {target_table} built successfully")
    return target_table

def build_r_from_r9(con, r, args):

    print(f"[R{r}] building from R9...")

    df = con.execute("""
        SELECT veh_id, t_bin, hex_id
        FROM trajectory_states_r9
    """).fetchdf()

    df["hex_id"] = [
        str(h3.cell_to_parent(h, r))
        for h in df["hex_id"]
    ]

    # collapse duplicates AFTER resolution change
    df = df.drop_duplicates(subset=["veh_id", "t_bin", "hex_id"])

    table_name = f"trajectory_states_r{r}"

    con.execute(f"DROP TABLE IF EXISTS {table_name}")

    table = pa.Table.from_pandas(df, preserve_index=False)
    con.register("tmp", table)

    con.execute(f"""
        CREATE TABLE {table_name} AS
        SELECT * FROM tmp
    """)

    con.unregister("tmp")

    print(f"[R{r}] built successfully")

def run_pipeline(args):

    print(f"203 {args}")

    con = duckdb.connect(args.output)
    con.execute("PRAGMA enable_object_cache=false")
    con.execute(f"ATTACH '{args.input}' AS input_db")

    r9_table = "trajectory_states_r9"
    target_table = f"trajectory_states_r{args.res}"
    metadata_table = f"state_metadata_r{args.res}"

    print(f"219 {args}")

    # ---------------------------------------------------
    # STEP 1: skip if final exists
    # ---------------------------------------------------
    if con.execute(f"""
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema='main'
          AND table_name='{target_table}'
    """).fetchone():

        print(f"[SKIP] {target_table} already exists")
        con.close()
        return

    print(f"232 {args}")

    # ---------------------------------------------------
    # STEP 2: ensure R9 exists
    # ---------------------------------------------------
    r9_exists = con.execute("""
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema='main'
          AND table_name='trajectory_states_r9'
    """).fetchone()

    if not r9_exists:
        print("[R9] building trajectory_states_r9")
        build_trajectory_states_r9(con, args)
    else:
        print("[R9] using existing trajectory_states_r9")

    print(f"250 {args}")

    # ---------------------------------------------------
    # STEP 3: ONLY H3 TRANSFORMATION (NO SQL QUERY SYSTEM)
    # ---------------------------------------------------
    if args.res == 9:
        print("[R9] target is 9 → nothing to derive")
        con.close()
        return

    print(f"255 {args}")

    df = con.execute("""
        SELECT veh_id, t_bin, hex_id
        FROM trajectory_states_r9
    """).fetchdf()

    print(f"264 {args}")

    df["hex_id"] = [
        str(h3.cell_to_parent(h, args.res))
        for h in df["hex_id"]
    ]

    df = df.drop_duplicates(subset=["veh_id", "t_bin", "hex_id"])

    print(f"273 {args}")

    con.execute(f"DROP TABLE IF EXISTS {target_table}")

    table = pa.Table.from_pandas(df, preserve_index=False)
    con.register("tmp", table)

    con.execute(f"""
        CREATE TABLE {target_table} AS
        SELECT * FROM tmp
    """)

    con.unregister("tmp")

    print(f"277 {args}")

    # ---------------------------------------------------
    # STEP 4: metadata
    # ---------------------------------------------------
    con.execute(f"""
        CREATE TABLE {metadata_table} AS
        SELECT
            {args.res} AS h3_resolution,
            {args.timestep} AS timestep_seconds,
            'hex_id' AS h3_column,
            'trajectory_states_r9' AS source_table
    """)

    print(f"309 {args}")

    # ---------------------------------------------------
    # STEP 5: summary
    # ---------------------------------------------------
    stats = con.execute(f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT (veh_id, t_bin, hex_id)) AS n_states
        FROM {target_table}
    """).fetchdf()

    print("\n[STATE SPACE SUMMARY]")
    print(stats)

    con.close()
    
def main():
    args = parse_args()
    run_pipeline(args)

if __name__ == "__main__":
	main()