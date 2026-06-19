# ---------------------------------------------------
# Mobility LDP Analysis Toolkit
# filter_database_by_vehid.py
#
# Author: Jay M. Appleton
# License: Apache-2.0
# ---------------------------------------------------

import duckdb
import argparse


def load_teleported(con, path):
    """
    Load vehicle IDs (one integer per line) into DuckDB.
    """
    con.execute("""
        CREATE OR REPLACE TABLE teleported (
            veh_id INTEGER
        )
    """)

    con.execute(f"""
        COPY teleported FROM '{path}'
        (DELIMITER '\n')
    """)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_db", required=True, help="Input DuckDB file")
    parser.add_argument("-f", "--filter_file", required=True, help="vehicles_to_filter.info")
    parser.add_argument("-o", "--output_db", default="berlin.filtered.trajectories.duckdb")

    args = parser.parse_args()

    # Connect to input DB
    con = duckdb.connect(args.input_db)

    print("Loading teleported vehicle IDs...")
    load_teleported(con, args.filter_file)

    print("Creating filtered database...")

    # Attach output DB
    con.execute(f"ATTACH '{args.output_db}' AS out_db (READ_WRITE);")

    # Copy filtered tables (adjust table name if needed)
    con.execute("""
        CREATE TABLE out_db.fcd_segmented AS
        SELECT f.*
        FROM fcd_segmented f
        LEFT JOIN teleported t
        ON f.veh_id = t.veh_id
        WHERE t.veh_id IS NULL
    """)

    print("Done. Filtered DB written to:", args.output_db)


if __name__ == "__main__":
    main()
