"""
Central configuration for the CNN-BiLSTM -> LightGBM pipeline and the
raw-feature LightGBM baseline, matching Section 3 of:

  "Do CNN-BiLSTM Embeddings Improve LightGBM for Network Security
   Classification? A Controlled Representation-Ablation Study on
   CICIDS2017 and CICDarknet2020"

Every hyperparameter here is taken directly from Table 2 / Section 3 of
the manuscript. Edit ONLY the paths and the two label-column settings
before your first run -- inspect_data.py will tell you what those
should be for your actual files.
"""

import os

# ---------------------------------------------------------------------------
# Paths -- point these at your own files/folders. CICIDS2017 is normally
# distributed as several per-day CSVs; put them all in one folder and the
# loader will concatenate them. CICDarknet2020 is normally one CSV.
# ---------------------------------------------------------------------------
CICIDS2017_DIR = "data/cicids2017"          # folder of *.csv files
CICDARKNET2020_CSV = "data/cicdarknet2020/CICDarknet.CSV"

RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# Label columns -- CICIDS2017 and CICDarknet2020 CSVs from different mirrors
# use slightly different header spellings/capitalization/leading spaces.
# Run `python inspect_data.py` first and set these to what it reports.
# ---------------------------------------------------------------------------
CICIDS2017_LABEL_COL = "Label"          # confirmed via inspect_data.py
CICDARKNET2020_LABEL_COL = "Label"      # confirmed: the 4-class Non-Tor/NonVPN/Tor/VPN column
                                         # (NOT "Label.1", which is the fine-grained app category)
# CICDarknet2020 files often ship a second, finer-grained label (application
# category) in a column such as "Label2" -- that one is NOT used here; only
# the 4 traffic-origin classes described in the paper are used.
CICDARKNET2020_CLASSES = ["Non-Tor", "NonVPN", "Tor", "VPN"]

# ---------------------------------------------------------------------------
# Pilot sampling (Section 3.1/3.2/Table 1)
# "class-aware pilot subsets": cap each class at a maximum number of rows
# (taking all available rows if a class has fewer than the cap), so common
# classes are downsampled and rare classes are kept whole. Exact totals
# (1,868 / 2,000 in the manuscript) depend on class composition of the
# specific CICIDS2017/CICDarknet2020 files used for that run and will NOT
# be reproduced exactly here -- see README.md.
# ---------------------------------------------------------------------------
CICIDS2017_MAX_PER_CLASS = 150   # tune to get close to a 1,868-row pilot
CICDARKNET2020_MAX_PER_CLASS = 500  # 500 * 4 classes = 2,000, matches Table 1

RANDOM_SEED = 42
TEST_SIZE = 0.20  # stratified 80:20 hold-out (Section 3.2)

# ---------------------------------------------------------------------------
# CNN-BiLSTM architecture + training (Table 2)
# ---------------------------------------------------------------------------
CNN_CH1 = 16          # Conv1D: 1 -> 16 channels, kernel=3, padding=1
CNN_CH2 = 32          # Conv1D: 16 -> 32 channels, kernel=3, padding=1
CNN_KERNEL = 3
CNN_PADDING = 1

LSTM_HIDDEN = 32       # BiLSTM hidden size per direction (input size 32)
LSTM_LAYERS = 1
LSTM_BIDIRECTIONAL = True

EMBEDDING_DIM = 64     # 64-dimensional dense embedding
DROPOUT = 0.15

OPTIMIZER_LR = 0.001   # Adam, learning rate = 0.001
NUM_EPOCHS = 30        # not specified exactly in the manuscript -- tune/report
BATCH_SIZE = 64        # not specified exactly in the manuscript -- tune/report

# ---------------------------------------------------------------------------
# LightGBM (Table 2 / Section 3.4)
# ---------------------------------------------------------------------------
LGBM_PARAMS = dict(
    n_estimators=120,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Bootstrap (Section 3.6)
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 5000
CI_LOW, CI_HIGH = 2.5, 97.5  # percentiles for the 95% CI

os.makedirs(RESULTS_DIR, exist_ok=True)
