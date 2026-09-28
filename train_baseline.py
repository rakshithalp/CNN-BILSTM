"""
Controlled raw-feature LightGBM baseline -- Section 3.4.

Same pilot data, same split, same seed, same LightGBM hyperparameters as
the hybrid, but trained directly on the standardized raw numeric features
(no CNN-BiLSTM stage).
"""

from lightgbm import LGBMClassifier

import config


def run_baseline_pipeline(X_train, X_test, y_train, y_test):
    print("  [baseline] fitting LightGBM directly on standardized raw features ...")
    clf = LGBMClassifier(**config.LGBM_PARAMS, verbosity=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, clf
