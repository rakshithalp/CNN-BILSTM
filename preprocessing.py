"""
Data loading, cleaning, class-aware pilot sampling, and preprocessing --
implements Section 3.1 / 3.2 of the manuscript:

  - non-target columns separated from the class label
  - only numeric traffic features used
  - infinite values -> NaN
  - all-missing / constant columns dropped
  - remaining missing values filled with the TRAINING partition's median
  - features standardized using TRAINING-partition statistics only
  - labels whitespace-normalized then numerically encoded
  - fixed seed (42), stratified 80:20 hold-out split
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config


def _clean_columns(df):
    df.columns = [c.strip() for c in df.columns]
    return df


def _normalize_cicids_labels(labels):
    """The public CICIDS2017 CSVs have the em-dash in 'Web Attack – X'
    baked in as the UTF-8 replacement character (U+FFFD) at the source --
    it is not recoverable via a different read encoding. Normalize any
    'Web Attack <junk> X' to 'Web Attack - X' so the three web-attack
    subtypes collapse into stable, matching class names instead of being
    3 separate garbled strings (or worse, 3 non-matching ones per file)."""
    out = []
    for lab in labels:
        m = re.match(r"^Web Attack\s*\S*\s*(.+)$", lab)
        if m and "Web Attack" in lab:
            out.append(f"Web Attack - {m.group(1).strip()}")
        else:
            out.append(lab)
    return out


def _class_aware_pilot_sample(df, label_col, max_per_class, seed):
    """Cap each class at `max_per_class` rows; keep all rows for classes
    with fewer than that. This is the 'class-aware pilot subset' strategy
    described in Section 3.2. Exact resulting totals depend on the class
    composition of your specific files."""
    rng = np.random.RandomState(seed)
    parts = []
    for cls, group in df.groupby(label_col):
        n = min(len(group), max_per_class)
        parts.append(group.sample(n=n, random_state=rng.randint(0, 2**31 - 1)))
    return pd.concat(parts, axis=0).reset_index(drop=True)


def _clean_features(X_train, X_test):
    """Infinite -> NaN, drop all-missing/constant columns (fit on train),
    median-impute using TRAIN statistics, standardize using TRAIN statistics."""
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    all_missing = X_train.columns[X_train.isna().all()]
    X_train = X_train.drop(columns=all_missing)
    X_test = X_test.drop(columns=all_missing, errors="ignore")

    constant_cols = X_train.columns[X_train.nunique(dropna=True) <= 1]
    X_train = X_train.drop(columns=constant_cols)
    X_test = X_test.drop(columns=constant_cols, errors="ignore")

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # any column in X_test missing entirely (edge case) gets the train median
    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = medians[c]
    X_test = X_test[X_train.columns]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values.astype(np.float64))
    X_test_scaled = scaler.transform(X_test.values.astype(np.float64))

    return X_train_scaled, X_test_scaled, list(X_train.columns)


def load_cicids2017():
    files = sorted(glob.glob(os.path.join(config.CICIDS2017_DIR, "*.csv"))) + \
        sorted(glob.glob(os.path.join(config.CICIDS2017_DIR, "*.CSV")))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {config.CICIDS2017_DIR}. "
            "Set CICIDS2017_DIR in config.py."
        )
    dfs = [_clean_columns(pd.read_csv(f, low_memory=False)) for f in files]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    label_col = config.CICIDS2017_LABEL_COL
    if label_col not in df.columns:
        raise KeyError(
            f"Label column '{label_col}' not found. Columns are: {list(df.columns)}. "
            "Run inspect_data.py and fix CICIDS2017_LABEL_COL in config.py."
        )

    df[label_col] = df[label_col].astype(str).str.strip()
    df[label_col] = _normalize_cicids_labels(df[label_col].values)
    df = df.dropna(subset=[label_col])

    pilot = _class_aware_pilot_sample(
        df, label_col, config.CICIDS2017_MAX_PER_CLASS, config.RANDOM_SEED
    )

    y_raw = pilot[label_col].values
    X = pilot.drop(columns=[label_col])
    X = X.select_dtypes(include=[np.number])  # numeric traffic features only

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )

    X_train, X_test, feature_names = _clean_features(X_train_df, X_test_df)
    return X_train, X_test, y_train, y_test, le.classes_, feature_names


def load_cicdarknet2020():
    path = config.CICDARKNET2020_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Set CICDARKNET2020_CSV in config.py."
        )
    df = _clean_columns(pd.read_csv(path, low_memory=False))

    label_col = config.CICDARKNET2020_LABEL_COL
    if label_col not in df.columns:
        raise KeyError(
            f"Label column '{label_col}' not found. Columns are: {list(df.columns)}. "
            "Run inspect_data.py and fix CICDARKNET2020_LABEL_COL in config.py."
        )

    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[label_col].isin(config.CICDARKNET2020_CLASSES)].copy()
    if df.empty:
        raise ValueError(
            f"No rows matched CICDARKNET2020_CLASSES={config.CICDARKNET2020_CLASSES} "
            f"in column '{label_col}'. Run inspect_data.py to see actual label values."
        )

    pilot = _class_aware_pilot_sample(
        df, label_col, config.CICDARKNET2020_MAX_PER_CLASS, config.RANDOM_SEED
    )

    y_raw = pilot[label_col].values
    X = pilot.drop(columns=[label_col])
    # drop any other label-ish column (e.g. a finer-grained "Label2") so it
    # can't leak into the numeric features
    X = X.drop(columns=[c for c in X.columns if "label" in c.lower()], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    le = LabelEncoder()
    le.fit(config.CICDARKNET2020_CLASSES)  # fixes class order to the paper's list
    y = le.transform(y_raw)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )

    X_train, X_test, feature_names = _clean_features(X_train_df, X_test_df)
    return X_train, X_test, y_train, y_test, le.classes_, feature_names
