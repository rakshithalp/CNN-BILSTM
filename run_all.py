"""
Runs the full controlled comparison on both datasets and writes out CSVs
matching every results table in the manuscript (Tables 3-9):

  results/<dataset>_table3_aggregate.csv       -- accuracy/precision/recall/F1, both models
  results/<dataset>_table4_hybrid_ci.csv       -- bootstrap 95% CI, hybrid
  results/<dataset>_table4b_baseline_ci.csv    -- bootstrap 95% CI, baseline (NEW -- paper only had this for hybrid)
  results/<dataset>_class_metrics.csv          -- Tables 5/6, per-class precision/recall/F1/support (hybrid)
  results/<dataset>_confusion_matrix.csv       -- Table 7 equivalent
  results/<dataset>_table9_ablation.csv        -- baseline vs hybrid delta
  results/<dataset>_paired_significance.csv    -- NEW: paired bootstrap test on the baseline-hybrid gap

Usage:
    python inspect_data.py         # first, to confirm label columns
    # edit config.py if needed
    python run_all.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)

import config
from bootstrap import bootstrap_ci, paired_bootstrap_diff
from preprocessing import load_cicdarknet2020, load_cicids2017
from train_baseline import run_baseline_pipeline
from train_hybrid import run_hybrid_pipeline


def aggregate_row(model_name, y_true, y_pred):
    return {
        "model": model_name,
        "accuracy_%": 100 * accuracy_score(y_true, y_pred),
        "macro_precision_%": 100 * precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall_%": 100 * recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1_%": 100 * f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def run_dataset(name, loader_fn):
    print("=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    X_train, X_test, y_train, y_test, class_names, feature_names = loader_fn()
    n_classes = len(class_names)
    print(f"  train={len(X_train)}  test={len(X_test)}  n_classes={n_classes}  "
          f"n_features={X_train.shape[1]}")

    # --- Hybrid: CNN-BiLSTM embedding + LightGBM -----------------------
    y_pred_hybrid, embedder, lgbm_hybrid = run_hybrid_pipeline(
        X_train, X_test, y_train, y_test, n_classes, name
    )

    # --- Baseline: raw standardized features + LightGBM -----------------
    y_pred_baseline, lgbm_baseline = run_baseline_pipeline(X_train, X_test, y_train, y_test)

    # --- Table 3: aggregate comparison -----------------------------------
    table3 = pd.DataFrame([
        aggregate_row("CNN-BiLSTM embedding + LightGBM", y_test, y_pred_hybrid),
        aggregate_row("Raw standardized features + LightGBM", y_test, y_pred_baseline),
    ])
    table3.to_csv(f"{config.RESULTS_DIR}/{name}_table3_aggregate.csv", index=False)
    print("\n  Table 3 (aggregate comparison):")
    print(table3.to_string(index=False))

    # --- Table 4: bootstrap CI, hybrid AND baseline -----------------------
    ci_hybrid = bootstrap_ci(y_test, y_pred_hybrid)
    ci_baseline = bootstrap_ci(y_test, y_pred_baseline)

    def ci_to_df(ci, label):
        rows = []
        for metric, v in ci.items():
            rows.append({
                "metric": metric,
                "point_estimate_%": 100 * v["point"],
                "ci_low_%": 100 * v["ci_low"],
                "ci_high_%": 100 * v["ci_high"],
            })
        df = pd.DataFrame(rows)
        df.insert(0, "model", label)
        return df

    ci_to_df(ci_hybrid, "hybrid").to_csv(
        f"{config.RESULTS_DIR}/{name}_table4_hybrid_ci.csv", index=False)
    ci_to_df(ci_baseline, "baseline").to_csv(
        f"{config.RESULTS_DIR}/{name}_table4b_baseline_ci.csv", index=False)
    print("\n  Table 4 (bootstrap 95% CI, hybrid):")
    print(ci_to_df(ci_hybrid, "hybrid").to_string(index=False))
    print("\n  Table 4b (bootstrap 95% CI, baseline -- not in original manuscript):")
    print(ci_to_df(ci_baseline, "baseline").to_string(index=False))

    # --- Tables 5/6: per-class metrics (hybrid) ---------------------------
    report = classification_report(
        y_test, y_pred_hybrid, target_names=[str(c) for c in class_names],
        output_dict=True, zero_division=0,
    )
    class_metrics = pd.DataFrame(report).T
    class_metrics.to_csv(f"{config.RESULTS_DIR}/{name}_class_metrics.csv")
    print("\n  Per-class metrics (hybrid):")
    print(class_metrics.to_string())

    # --- Table 7: confusion matrix -----------------------------------------
    cm = confusion_matrix(y_test, y_pred_hybrid)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f"{config.RESULTS_DIR}/{name}_confusion_matrix.csv")
    print("\n  Confusion matrix (hybrid):")
    print(cm_df.to_string())

    # --- Table 9: representation ablation -----------------------------------
    acc_h = 100 * accuracy_score(y_test, y_pred_hybrid)
    acc_b = 100 * accuracy_score(y_test, y_pred_baseline)
    f1_h = 100 * f1_score(y_test, y_pred_hybrid, average="macro", zero_division=0)
    f1_b = 100 * f1_score(y_test, y_pred_baseline, average="macro", zero_division=0)
    table9 = pd.DataFrame([{
        "dataset": name,
        "baseline_accuracy_%": acc_b,
        "hybrid_accuracy_%": acc_h,
        "delta_accuracy_pp": acc_h - acc_b,
        "baseline_macro_f1_%": f1_b,
        "hybrid_macro_f1_%": f1_h,
        "delta_macro_f1_pp": f1_h - f1_b,
    }])
    table9.to_csv(f"{config.RESULTS_DIR}/{name}_table9_ablation.csv", index=False)
    print("\n  Table 9 (representation ablation):")
    print(table9.to_string(index=False))

    # --- Paired bootstrap significance test (fixes the gap the paper flags) -
    paired = paired_bootstrap_diff(y_test, y_pred_hybrid, y_pred_baseline)
    paired_rows = []
    for metric, v in paired.items():
        paired_rows.append({
            "metric": metric,
            "point_diff_hybrid_minus_baseline_%": 100 * v["point_diff"],
            "ci_low_%": 100 * v["ci_low"],
            "ci_high_%": 100 * v["ci_high"],
            "significant_at_95%": v["significant_95"],
        })
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(f"{config.RESULTS_DIR}/{name}_paired_significance.csv", index=False)
    print("\n  Paired bootstrap significance test (hybrid - baseline):")
    print(paired_df.to_string(index=False))
    print()

    return table3, table9


if __name__ == "__main__":
    print(f"Random seed: {config.RANDOM_SEED}")
    print(f"Results will be written to: {config.RESULTS_DIR}/\n")

    run_dataset("CICIDS2017", load_cicids2017)
    run_dataset("CICDarknet2020", load_cicdarknet2020)

    print("=" * 70)
    print("Done. All tables written to the results/ folder.")
    print("=" * 70)
