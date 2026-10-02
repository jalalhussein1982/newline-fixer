"""The from-scratch gap tagger: word embedding + character CNN + BiLSTM (design 4.4)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .scratch_config import ScratchConfig
from .vocab import NO_GAP, PAD, Batch

N_CLASSES = 4


class CharCNN(nn.Module):
    def __init__(self, n_chars: int, char_dim: int, filters: int, width: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_chars, char_dim, padding_idx=PAD)
        self.conv = nn.Conv1d(char_dim, filters, width, padding=width // 2)

    def forward(self, char_ids: Tensor) -> Tensor:
        b, n, c = char_ids.shape
        x = self.emb(char_ids.reshape(b * n, c)).transpose(1, 2)  # [B*N, dim, C]
        x = torch.relu(self.conv(x))  # [B*N, filters, C]
        return x.max(dim=2).values.reshape(b, n, -1)


class GapTagger(nn.Module):
    def __init__(self, cfg: ScratchConfig, n_words: int, n_chars: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.word_emb = nn.Embedding(n_words, cfg.word_dim, padding_idx=PAD)
        self.char_cnn = CharCNN(n_chars, cfg.char_dim, cfg.char_filters, cfg.char_width)
        self.gap_emb = nn.Embedding(NO_GAP + 1, cfg.gap_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.lstm = nn.LSTM(
            cfg.word_dim + cfg.char_filters + cfg.gap_dim,
            cfg.hidden,
            num_layers=cfg.layers,
            bidirectional=True,
            batch_first=True,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(4 * cfg.hidden + cfg.gap_dim, cfg.classifier_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.classifier_hidden, N_CLASSES),
        )

    def forward(self, batch: Batch) -> Tensor:
        gap = self.gap_emb(batch.gap_after)  # [B, N, gap_dim]
        x = torch.cat([self.word_emb(batch.word), self.char_cnn(batch.char), gap], dim=-1)
        x = self.drop(x)
        packed = pack_padded_sequence(x, batch.lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(out, batch_first=True, total_length=batch.word.shape[1])
        left, right = h[:, :-1], h[:, 1:]  # hidden states on both sides of each gap
        feats = torch.cat([left, right, gap[:, :-1]], dim=-1)
        logits: Tensor = self.head(feats)
        return logits


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
