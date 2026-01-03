#!/usr/bin/env python3
"""
Benchmark runner automation script for Onyx.
"""

import subprocess
import sys

def main():
    print("Building Onyx in Release mode...")
    subprocess.run(["make", "build"], check=True)

    print("\nExecuting Onyx Performance Benchmarks:\n")
    res = subprocess.run(["./build/onyx-bench"])
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
