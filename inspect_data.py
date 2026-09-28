"""
Run this FIRST, before anything else.

It loads just the headers (and a peek at label values) from your
CICIDS2017 folder and CICDarknet2020 file, so you can set the correct
column names in config.py (CICIDS2017_LABEL_COL, CICDARKNET2020_LABEL_COL,
CICDARKNET2020_CLASSES) before running the real pipeline.

Different mirrors of these datasets ship with different header spelling,
capitalization, or leading/trailing spaces (e.g. " Label" vs "Label"),
which is why this step exists rather than hardcoding a guess.
"""

import glob
import os

import pandas as pd

import config


def inspect_cicids2017():
    print("=" * 70)
    print("CICIDS2017 —", config.CICIDS2017_DIR)
    print("=" * 70)
    files = sorted(glob.glob(os.path.join(config.CICIDS2017_DIR, "*.csv"))) + \
        sorted(glob.glob(os.path.join(config.CICIDS2017_DIR, "*.CSV")))
    if not files:
        print(f"  No CSV files found in {config.CICIDS2017_DIR} -- set CICIDS2017_DIR in config.py")
        return
    print(f"  Found {len(files)} file(s), reading header + 2000 rows of the first one:")
    df = pd.read_csv(files[0], nrows=2000, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print("  Columns:", list(df.columns))
    label_like = [c for c in df.columns if "label" in c.lower()]
    print("  Column(s) containing 'label':", label_like)
    for c in label_like:
        print(f"    Unique values in '{c}' (first 2000 rows):", df[c].unique()[:20])


def inspect_cicdarknet2020():
    print()
    print("=" * 70)
    print("CICDarknet2020 —", config.CICDARKNET2020_CSV)
    print("=" * 70)
    if not os.path.exists(config.CICDARKNET2020_CSV):
        print(f"  File not found -- set CICDARKNET2020_CSV in config.py")
        return
    df = pd.read_csv(config.CICDARKNET2020_CSV, nrows=2000, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print("  Columns:", list(df.columns))
    label_like = [c for c in df.columns if "label" in c.lower()]
    print("  Column(s) containing 'label':", label_like)
    for c in label_like:
        print(f"    Unique values in '{c}' (first 2000 rows):", df[c].unique()[:20])


if __name__ == "__main__":
    inspect_cicids2017()
    inspect_cicdarknet2020()
    print()
    print("Now edit config.py: set CICIDS2017_LABEL_COL, CICDARKNET2020_LABEL_COL,")
    print("and CICDARKNET2020_CLASSES to match what was printed above.")
