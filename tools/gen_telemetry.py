#!/usr/bin/env python3
"""
Synthetic Telemetry Generator for Onyx Benchmarking
Generates realistic high-frequency telemetry records (CSV format).
"""

import sys
import random
import time

def generate_telemetry(num_records: int, output_file: str):
    print(f"Generating {num_records:,} telemetry records into {output_file}...")
    start = time.time()

    with open(output_file, "w") as f:
        f.write("device_id,latency_us,temperature,status,timestamp\n")
        base_ts = 1700000000000000

        for i in range(num_records):
            dev_id = random.randint(100, 1000)
            latency = round(random.expovariate(1.0 / 250.0), 2)
            temp = round(random.normalvariate(55.0, 8.0), 2)
            status = 1 if latency < 1000 else (2 if latency < 2500 else 3)
            ts = base_ts + i * 100

            f.write(f"{dev_id},{latency},{temp},{status},{ts}\n")

    elapsed = time.time() - start
    print(f"Done in {elapsed:.2f}s ({num_records / elapsed:,.0f} records/sec)")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    out = sys.argv[2] if len(sys.argv) > 2 else "telemetry.csv"
    generate_telemetry(count, out)
