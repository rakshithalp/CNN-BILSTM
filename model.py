"""
CNN-BiLSTM representation learner -- Section 3.3 / Table 2.

Architecture, exactly as described:
  Conv1D(1->16, k=3, pad=1) -> ReLU -> MaxPool
  Conv1D(16->32, k=3, pad=1) -> ReLU -> MaxPool
  transpose to sequence form -> BiLSTM(input=32, hidden=32, 1 layer, bidirectional)
  mean-pool over the sequence -> FC(64 -> 64) -> ReLU -> Dropout(0.15)
  auxiliary linear classifier head (used only during training)
"""

import torch
import torch.nn as nn

import config


class CNNBiLSTMEmbedder(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, config.CNN_CH1, kernel_size=config.CNN_KERNEL,
                                padding=config.CNN_PADDING)
        self.conv2 = nn.Conv1d(config.CNN_CH1, config.CNN_CH2, kernel_size=config.CNN_KERNEL,
                                padding=config.CNN_PADDING)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(
            input_size=config.CNN_CH2,
            hidden_size=config.LSTM_HIDDEN,
            num_layers=config.LSTM_LAYERS,
            bidirectional=config.LSTM_BIDIRECTIONAL,
            batch_first=True,
        )

        lstm_out_dim = config.LSTM_HIDDEN * (2 if config.LSTM_BIDIRECTIONAL else 1)  # 64
        self.fc_embed = nn.Linear(lstm_out_dim, config.EMBEDDING_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)

        self.classifier = nn.Linear(config.EMBEDDING_DIM, n_classes)  # auxiliary head only

    def embed(self, x):
        # x: (batch, n_features) -> (batch, 1, n_features) for Conv1d
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # x: (batch, 32, seq_len) -> (batch, seq_len, 32) for the LSTM
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, 64)
        pooled = lstm_out.mean(dim=1)       # mean-pool across the sequence
        emb = self.relu(self.fc_embed(pooled))
        emb = self.dropout(emb)
        return emb

    def forward(self, x):
        emb = self.embed(x)
        logits = self.classifier(emb)
        return logits, emb


def compute_class_weights(y, n_classes, device):
    counts = torch.bincount(torch.as_tensor(y), minlength=n_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (n_classes * counts)
    return weights.to(device)
