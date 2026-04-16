from __future__ import annotations

import torch
from torch import nn


ModelConfig = dict[str, int | float]


class SudokuMLP(nn.Module):
    def __init__(
        self,
        embed_dim: int = 16,
        hidden_dim: int = 512,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout

        self.embedding = nn.Embedding(10, embed_dim)

        layers = []
        input_dim = 81 * (embed_dim + 1)
        current_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(current_dim, 81 * 9)

    def forward(self, digits: torch.Tensor, givens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(digits)
        features = torch.cat((embedded, givens.unsqueeze(-1)), dim=-1)
        flattened = features.reshape(features.size(0), -1)
        logits = self.head(self.backbone(flattened))
        return logits.view(-1, 81, 9)

    def get_config(self) -> ModelConfig:
        return {
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "dropout": self.dropout,
        }


class SudokuTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        depth: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(10, embed_dim)
        self.givens_projection = nn.Linear(1, embed_dim)
        self.row_embedding = nn.Embedding(9, embed_dim)
        self.col_embedding = nn.Embedding(9, embed_dim)
        self.box_embedding = nn.Embedding(9, embed_dim)
        self.input_dropout = nn.Dropout(dropout)

        row_indices = torch.arange(81) // 9
        col_indices = torch.arange(81) % 9
        box_indices = (row_indices // 3) * 3 + (col_indices // 3)
        self.register_buffer("row_indices", row_indices, persistent=False)
        self.register_buffer("col_indices", col_indices, persistent=False)
        self.register_buffer("box_indices", box_indices, persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, 9)

    def forward(self, digits: torch.Tensor, givens: torch.Tensor) -> torch.Tensor:
        token_features = self.embedding(digits)
        token_features = token_features + self.givens_projection(givens.unsqueeze(-1))
        token_features = token_features + self.row_embedding(self.row_indices).unsqueeze(0)
        token_features = token_features + self.col_embedding(self.col_indices).unsqueeze(0)
        token_features = token_features + self.box_embedding(self.box_indices).unsqueeze(0)
        encoded = self.encoder(self.input_dropout(token_features))
        return self.head(encoded)

    def get_config(self) -> ModelConfig:
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "depth": self.depth,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        }


def create_model(model_type: str, **model_config: int | float) -> nn.Module:
    if model_type == "mlp":
        return SudokuMLP(**model_config)
    if model_type == "transformer":
        return SudokuTransformer(**model_config)
    raise ValueError(f"Unsupported model_type: {model_type}")
