# CNN-BiLSTM → LightGBM vs. Raw-Feature LightGBM: reproducible pipeline

Implements Section 3 of "Do CNN-BiLSTM Embeddings Improve LightGBM for
Network Security Classification?" end to end: preprocessing, class-aware
pilot sampling, the CNN-BiLSTM embedder, the LightGBM stage, the raw-feature
baseline, and bootstrap confidence intervals.

## This has already been run once, on your actual files

Your CICIDS2017 (8 daily CSVs) and CICDarknet2020 CSV were used to run this
end to end. Results are in `results/`. Key numbers from that run:

| Dataset | Model | Accuracy | Macro F1 |
|---|---|---|---|
| CICIDS2017 | Hybrid (CNN-BiLSTM+LightGBM) | 86.63% | 82.70% |
| CICIDS2017 | Baseline (raw+LightGBM) | 91.44% | 91.97% |
| CICDarknet2020 | Hybrid | 79.75% | 79.83% |
| CICDarknet2020 | Baseline | 94.25% | 94.29% |

**This confirms the paper's central claim, on your real data, with a fresh
run**: the raw-feature baseline beats the hybrid on both datasets. The paired
bootstrap significance test (new — the manuscript didn't have this because
per-sample predictions from the original run weren't archived) shows the gap
is statistically significant at 95% on both accuracy and macro-F1, both
datasets (`results/*_paired_significance.csv`).

**These numbers are NOT identical to the manuscript's** (84.49%/82.85% and
75.25%/74.92% for the hybrid). That is expected, not a bug — see "Why the
numbers differ" below. If you want to update the paper, use the numbers in
`results/`, not the manuscript's current placeholder numbers, since these
came from your real files rather than a lost/unrecorded run.

## How the pipeline maps onto the manuscript

| File | Manuscript section |
|---|---|
| `preprocessing.py` | 3.1 Datasets, 3.2 Preprocessing (cleaning, pilot sampling, split) |
| `model.py` | 3.3, Table 2 (CNN-BiLSTM architecture) |
| `train_hybrid.py` | 3.3 (embedding extraction + LightGBM on embeddings) |
| `train_baseline.py` | 3.4 (raw-feature LightGBM baseline) |
| `bootstrap.py` | 3.6 Statistical Analysis |
| `run_all.py` | orchestrates everything, writes Tables 3-9 equivalents |

## Running it yourself (e.g. to try different settings)

```bash
pip install torch lightgbm pandas scikit-learn numpy
python inspect_data.py   # confirms label column names/values for your files
python run_all.py        # writes results/ CSVs
```

Dataset paths, the pilot-sampling cap, and every model/LightGBM
hyperparameter are in `config.py`, with the manuscript section each one
comes from noted in a comment.

## Why the numbers differ from the manuscript

The manuscript states results were computed on "class-aware pilot subsets"
but the exact sampling rule was never pinned down in writing beyond
"class-aware." We reverse-engineered it from the reported sample sizes:

- **CICIDS2017**: capping every class at 150 rows (taking all rows for a
  class with fewer than 150) gives exactly 1,868 total rows — matching
  Table 1 exactly. `CICIDS2017_MAX_PER_CLASS = 150` in `config.py`.
- **CICDarknet2020**: capping every class at 500 rows gives exactly 2,000 —
  also matching Table 1 exactly, and matching the per-class test support of
  100 in Table 6. `CICDARKNET2020_MAX_PER_CLASS = 500`.

So the *sample sizes* match the manuscript exactly. The *specific rows*
drawn for the "common" classes (the ones over the cap) still depend on:

1. **Which random draw** — `np.random.RandomState(42)` produces a
   *specific* pseudorandom sequence, but the exact rows it picks depend on
   the exact order operations are called in (e.g. `groupby` iteration
   order, library versions). There is no way to guarantee bit-for-bit
   identical row selection to a lost original script without that script.
2. **Neural training stochasticity** — even with `torch.manual_seed(42)`,
   full determinism across PyTorch versions/CPU vs GPU is not guaranteed
   without additional flags (`torch.use_deterministic_algorithms`), which
   were not specified in the manuscript.
3. **Training duration** — the manuscript doesn't state the number of
   epochs or batch size for the CNN-BiLSTM (`config.py` uses 30 epochs,
   batch size 64, as reasonable defaults — tune these and re-run if you
   want to try to close the gap further, and report whatever you settle on
   in a revised Methods section).

This is exactly the situation flagged in the manuscript's own Section 5.3
and in the earlier review of this paper: a single run at pilot scale isn't
reproducible down to the decimal point, only in **direction and rough
magnitude** — which it does reproduce here.

## What this does NOT fix

- Still pilot-scale data (by design, matching the manuscript's stated
  scope), not the full CICIDS2017 (2.83M rows) / CICDarknet2020 (158K rows).
- Still a single train/test split and a single seed. To do the "five
  independent seeds/folds" the manuscript's own Section 5.3 calls for,
  wrap `run_dataset()` in a loop over seeds and aggregate — not implemented
  here since that's a substantive rerun decision, not a code-completeness
  gap, and would take considerably longer to execute.
- The web-attack label corruption fix (`_normalize_cicids_labels` in
  `preprocessing.py`) is worth a one-line mention in your Methods section
  if you use these numbers — reviewers familiar with CICIDS2017 will
  recognize the mojibake issue and may ask how it was handled.
