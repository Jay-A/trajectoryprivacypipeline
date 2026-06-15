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
        epilog=USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter )

    # parser = argparse.ArgumentParser(
    #     description="Build LDP-ready spatiotemporal state space from H3 trajectory data" )

    parser.add_argument( "-i", "--input", required=True,
        help="Input raw DuckDB file  " )

    parser.add_argument( "-o", "--output", default="statecount.duckdb",
        help="Output DuckDB file for state space" )

    parser.add_argument( "-t", "--timestep", type=int, default=30,
        help="Time bin size in seconds (default: 30)" )

    parser.add_argument( "--source-table", default="trajectory_states_r9",
        help="Input table name inside DuckDB" )

    parser.add_argument("--res", type=int, default=9,
        help="H3 resolution for state space (default: 9)")

    # parser.add_argument( "--h3-column", default="h3_9",
    #     help="Column containing H3 index" )

    parser.add_argument( "--time-column", default="time",
        help="Column containing timestamp" )

    parser.add_argument( "--overwrite", action="store_true",
        help="Drop existing output tables before creating new ones" )

    parser.add_argument(
        "--agg", type=str, default="last", choices=["last", "max"],
        help="State collapse strategy within (veh_id, t_bin): 'last' (temporal) or 'max' (most frequent)"
    )

    return parser.parse_args()

def ensure_trajectory_states_r9(con):

    tables = con.execute("SHOW TABLES FROM input_db").fetchdf()["name"].tolist()

    if "trajectory_states_r9" in tables:
        con.execute("DROP TABLE input_db.trajectory_states_r9")

    print("Building trajectory_states_r9 from fcd_segmented...")

    import pyproj
    import h3
    import pyarrow as pa

    # --------------------------------------------------
    # 0. LOAD RAW SUMO DATA (local coords)
    # --------------------------------------------------
    df = con.execute("""
        SELECT veh_id, time, x, y, speed
        FROM input_db.fcd_segmented
    """).fetchdf()

    # --------------------------------------------------
    # 1. APPLY NET OFFSET (CRITICAL — FIXES "OCEAN BUG")
    # --------------------------------------------------
    NET_X = 363050.47
    NET_Y = 5798536.79

    df["x"] = df["x"] + NET_X
    df["y"] = df["y"] + NET_Y

    # --------------------------------------------------
    # 2. UTM → WGS84
    # --------------------------------------------------
    to_latlon = pyproj.Transformer.from_crs(
        "EPSG:32633",
        "EPSG:4326",
        always_xy=True
    )

    lon, lat = to_latlon.transform(df["x"].values, df["y"].values)

    # --------------------------------------------------
    # 3. H3 encoding (r9 canonical)
    # --------------------------------------------------
    df["hex_id"] = [
        str(h3.latlng_to_cell(lat_i, lon_i, 9))
        for lat_i, lon_i in zip(lat, lon)
    ]

    # enforce clean schema (prevents STRUCT bug in DuckDB)
    df["hex_id"] = df["hex_id"].astype("string")

    df = df[["veh_id", "time", "x", "y", "speed", "hex_id"]]

    table = pa.Table.from_pandas(df, preserve_index=False)

    con.register("tmp", table)

    con.execute("""
        CREATE TABLE input_db.trajectory_states_r9 AS
        SELECT * FROM tmp
    """)

    con.unregister("tmp")

    return "trajectory_states_r9"

def ensure_h3_table(con, res):
    # table_name = f"traj_h3_res{res}"

    existing = con.execute("SHOW TABLES FROM input_db").fetchdf()["name"].tolist()
    if table_name in existing:
        return table_name

    if res != 9:
        raise ValueError("Only r=9 base H3 construction is supported from raw data")

    print("Building H3 r=9 table from fcd_segmented...")

    con.execute(f"""
        CREATE TABLE input_db.{table_name} AS
        SELECT
            veh_id,
            time,
            x,
            y,
            speed,
            h3_latlng_to_cell(y, x, 9) AS h3_9
        FROM input_db.fcd_segmented
    """)

    return table_name    

def build_state_query(args, table_name, source_table, h3_col):

    base = f"""
        SELECT
            veh_id,
            CAST(FLOOR(CAST({args.time_column} AS DOUBLE) / {args.timestep}) AS INTEGER) AS t_bin,
            {h3_col} AS hex_id,
            {args.time_column} AS time
        FROM input_db.{source_table}
    """

    if args.agg == "last":

        return f"""
        CREATE TABLE {table_name} AS
        SELECT veh_id, t_bin, hex_id
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY veh_id, t_bin
                       ORDER BY time DESC
                   ) AS rn
            FROM ({base})
        )
        WHERE rn = 1
        """

    elif args.agg == "max":

        return f"""
        CREATE TABLE {table_name} AS
        WITH base AS ({base}),
        counts AS (
            SELECT veh_id, t_bin, hex_id, COUNT(*) AS c
            FROM base
            GROUP BY veh_id, t_bin, hex_id
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY veh_id, t_bin
                       ORDER BY c DESC
                   ) AS rn
            FROM counts
        )
        SELECT veh_id, t_bin, hex_id
        FROM ranked
        WHERE rn = 1
        """

    else:
        raise ValueError(f"Unknown aggregation mode: {args.agg}")

def run_pipeline(args):

    con = duckdb.connect(args.output)

    # --------------------------------------------------
    # 1. ATTACH FIRST (critical fix)
    # --------------------------------------------------
    con.execute(f"ATTACH '{args.input}' AS input_db")

    # --------------------------------------------------
    # 2. Ensure canonical spatial layer exists (r9)
    # --------------------------------------------------
    base_table = ensure_trajectory_states_r9(con)

    # --------------------------------------------------
    # 3. State table naming
    # --------------------------------------------------
    table_name = f"trajectory_states_r{args.res}"
    metadata_table = f"state_metadata_r{args.res}"

    # --------------------------------------------------
    # 4. overwrite safety
    # --------------------------------------------------
    if args.overwrite:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"DROP TABLE IF EXISTS {metadata_table}")

    # --------------------------------------------------
    # 5. NO schema probing needed anymore
    #    (base is guaranteed)
    # --------------------------------------------------

    # --------------------------------------------------
    # 6. FIXED state query inputs
    # --------------------------------------------------
    query = build_state_query(args, table_name, base_table, "hex_id")
    con.execute(query)

    # --------------------------------------------------
    # 7b. SANITY CHECK: verify final state table schema
    # --------------------------------------------------
    schema_df = con.execute(f"""
        DESCRIBE {table_name}
    """).fetchdf()

    hex_type = schema_df.loc[
        schema_df["column_name"] == "hex_id",
        "column_type"
    ].iloc[0]

    if "STRUCT" in hex_type:
        raise ValueError(f"STATE TABLE CORRUPTED: hex_id is {hex_type}")

    if hex_type != "VARCHAR":
        raise ValueError(f"Unexpected hex_id type: {hex_type}")

    # --------------------------------------------------
    # 7. metadata
    # --------------------------------------------------
    con.execute(f"""
        CREATE TABLE {metadata_table} AS
        SELECT
            {args.res} AS h3_resolution,
            ? AS timestep_seconds,
            'hex_id' AS h3_column,
            ? AS time_column,
            ? AS source_table
    """, [
        args.timestep,
        args.time_column,
        base_table
    ])

    # --------------------------------------------------
    # 8. sanity stats
    # --------------------------------------------------
    stats = con.execute(f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT (veh_id, t_bin, hex_id)) AS n_states
        FROM {table_name}
    """).fetchdf()

    print("\n[STATE SPACE SUMMARY]")
    print(stats)

    con.close()
def main():
	args = parse_args()
	run_pipeline(args)

if __name__ == "__main__":
	main()