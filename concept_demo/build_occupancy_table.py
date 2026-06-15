import argparse
import duckdb

# - CLI ------------------------------------------------------
def parse_args():
    USAGE = """
    Example usage:

    python3 build_occupancy_table.py -i berlin.study.duckdb -r 8 
    """
    parser = argparse.ArgumentParser(
        description="Build occupancy tables from trajectory states" )

    parser.add_argument(
        "-i", "--input", required=True, help="Path to DuckDB database" )

    parser.add_argument(
        "-r", "--res", type=int, required=True, help="H3 resolution (e.g. 7, 9)" )

    parser.add_argument(
        "--drop-existing", action="store_true",
        help="Drop existing occupancy table before rebuilding" )

    parser.add_argument(
        "--dry-run", action="store_true", help="Print SQL instead of executing" )

    return parser.parse_args()

# - Core builder ----------------------------------------------
def build_occupancy(con, state_table, occupancy_table, args):

    if args.drop_existing:
        con.execute(f"DROP TABLE IF EXISTS {occupancy_table}")

    query = f"""
    CREATE TABLE {occupancy_table} AS
    SELECT
        hex_id,
        t_bin,
        COUNT(*) AS count
    FROM {state_table}
    GROUP BY hex_id, t_bin
    ORDER BY t_bin, hex_id
    """

    if args.dry_run:
        print(query)
        return

    con.execute(query)

# - Pipeline runner ------------------------------------------
def run_pipeline(args):

    con = duckdb.connect(args.input)

    state_table = f"trajectory_states_r{args.res}"
    occupancy_table = f"occupancy_r{args.res}"

    build_occupancy(con, state_table, occupancy_table, args)

    print(f"[OK] Built {occupancy_table} from {state_table}")

# - Main -----------------------------------------------------
def main():
    args = parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
