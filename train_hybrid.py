"""
Trains the CNN-BiLSTM embedder (with an auxiliary classifier head) on the
training partition, extracts 64-D embeddings for train and test, then fits
LightGBM on the training embeddings and evaluates on the test embeddings.

This is the "hybrid" pipeline (Section 3.3): standardized raw features ->
CNN-BiLSTM -> 64-D embedding -> LightGBM.
"""

import numpy as np
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from torch.utils.data import DataLoader, TensorDataset

import config
from model import CNNBiLSTMEmbedder, compute_class_weights


def train_embedder(X_train, y_train, n_classes, device):
    model = CNNBiLSTMEmbedder(n_features=X_train.shape[1], n_classes=n_classes).to(device)
    class_weights = compute_class_weights(y_train, n_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.OPTIMIZER_LR)

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(config.NUM_EPOCHS):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(ds)
        print(f"    epoch {epoch + 1}/{config.NUM_EPOCHS}  loss={avg_loss:.4f}")

    return model


@torch.no_grad()
def extract_embeddings(model, X, device, batch_size=512):
    model.eval()
    embeddings = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
        emb = model.embed(xb)
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def run_hybrid_pipeline(X_train, X_test, y_train, y_test, n_classes, dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [hybrid] training CNN-BiLSTM embedder on {device} ...")
    torch.manual_seed(config.RANDOM_SEED)

    embedder = train_embedder(X_train, y_train, n_classes, device)

    print("  [hybrid] extracting embeddings ...")
    emb_train = extract_embeddings(embedder, X_train, device)
    emb_test = extract_embeddings(embedder, X_test, device)

    print("  [hybrid] fitting LightGBM on embeddings ...")
    clf = LGBMClassifier(**config.LGBM_PARAMS, verbosity=-1)
    clf.fit(emb_train, y_train)
    y_pred = clf.predict(emb_test)

    return y_pred, embedder, clf
