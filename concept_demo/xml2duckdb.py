# ---------------------------------------------------
# Mobility LDP Analysis Toolkit
# xml2duckdb.py
#
# Author: Jay M. Appleton
# License: Apache-2.0
# ---------------------------------------------------

import argparse
import duckdb
import xml.etree.ElementTree as ET
import pyarrow as pa


# ---- Batch insert helper ----

def flush_batch(con, table, batch):
    if not batch:
        return

    arrays = {
        "veh_id": pa.array([r[0] for r in batch], type=pa.int32()),
        "time": pa.array([r[1] for r in batch], type=pa.float64()),
        "x": pa.array([r[2] for r in batch], type=pa.float64()),
        "y": pa.array([r[3] for r in batch], type=pa.float64()),
        "speed": pa.array([r[4] for r in batch], type=pa.float64()),
    }

    table_arrow = pa.Table.from_pydict(arrays)

    con.register("arrow_batch", table_arrow)
    con.execute(f"INSERT INTO {table} SELECT * FROM arrow_batch")
    con.unregister("arrow_batch")


# ---- Main pipeline ----

def main():
    parser = argparse.ArgumentParser(description="SUMO FCD → DuckDB pipeline (base only)")
    parser.add_argument("-i", "--input", required=True, help="Input SUMO FCD XML file")
    parser.add_argument("-o", "--output", default="base.duckdb", help="Output DuckDB file")

    args = parser.parse_args()

    con = duckdb.connect(args.output)

    # ---- Create raw table ----
    con.execute("""
        CREATE OR REPLACE TABLE fcd (
            veh_id INTEGER,
            time DOUBLE,
            x DOUBLE,
            y DOUBLE,
            speed DOUBLE
        )
    """)

    batch = []
    BATCH_SIZE = 100_000

    print("Parsing XML...")

    # ---- Stream XML ----
    for _, elem in ET.iterparse(args.input, events=("end",)):

        if elem.tag == "timestep":
            t = float(elem.attrib["time"])

            for v in elem.findall("vehicle"):
                batch.append((
                    int(v.attrib["id"]),
                    t,
                    float(v.attrib["x"]),
                    float(v.attrib["y"]),
                    float(v.attrib["speed"])
                ))

            if len(batch) >= BATCH_SIZE:
                flush_batch(con, "fcd", batch)
                batch.clear()

            elem.clear()

    # flush remaining
    flush_batch(con, "fcd", batch)

    print("Creating segmented trajectories...")

    con.execute("""
    CREATE OR REPLACE TABLE fcd_segmented AS
    WITH ordered AS (
        SELECT
            veh_id,
            time,
            x,
            y,
            speed,
            LAG(time) OVER w AS prev_time,
            LAG(x) OVER w AS prev_x,
            LAG(y) OVER w AS prev_y
        FROM fcd
        WINDOW w AS (PARTITION BY veh_id ORDER BY time)
    ),

    dist_calc AS (
        SELECT *,
            CASE
                WHEN prev_x IS NULL OR prev_y IS NULL THEN 0
                ELSE SQRT(
                    (x - prev_x) * (x - prev_x) +
                    (y - prev_y) * (y - prev_y)
                )
            END AS dist
        FROM ordered
    ),

    flagged AS (
        SELECT *,
            CASE
                WHEN prev_time IS NULL THEN 0
                WHEN (time - prev_time) > 1 THEN 1
                WHEN dist > 80 THEN 1
                ELSE 0
            END AS is_break
        FROM dist_calc
    )

    SELECT
        veh_id,
        time,
        x,
        y,
        speed,
        COALESCE(
            CAST(
                SUM(is_break) OVER (
                    PARTITION BY veh_id
                    ORDER BY time
                ) AS INTEGER
            ),
            0
        ) AS segment_id
    FROM flagged
    """)

    print("Done. Base DuckDB created.")

if __name__ == "__main__":
    main()