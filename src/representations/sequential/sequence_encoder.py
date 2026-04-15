from __future__ import annotations

from ..base import TrafficEncoder
import torch.nn as nn


class SequenceTrafficEncoder(TrafficEncoder):
    def __init__(self, vocab_size, d_model=64, n_heads=2, n_layers=2):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, x):
        x = self.token_emb(x)
        z = self.encoder(x)
        return z.mean(dim=1)  # pooling