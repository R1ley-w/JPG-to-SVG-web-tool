"""Autoregressive raster-to-SVG model.

A pretrained ResNet encoder produces an image embedding; a Transformer
decoder autoregressively emits SVG tokens (see :mod:`im2vec.tokenizer`).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv

from .tokenizer import Vocab

BACKBONES = {
    "resnet18": tv.resnet18,
    "resnet34": tv.resnet34,
    "resnet50": tv.resnet50,
}

BACKBONE_WEIGHTS = {
    "resnet18": tv.ResNet18_Weights,
    "resnet34": tv.ResNet34_Weights,
    "resnet50": tv.ResNet50_Weights,
}

# Feature channels of the final conv layer (before avgpool/fc).
BACKBONE_FEATURES = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}


class ImageEncoder(nn.Module):
    """ResNet backbone -> global image embedding of size ``d_model``."""

    def __init__(
        self,
        d_model: int = 512,
        backbone: str = "resnet18",
        pretrained: bool = True,
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r} (choose from {list(BACKBONES)})")

        weights = BACKBONE_WEIGHTS[backbone].DEFAULT if pretrained else None
        resnet = BACKBONES[backbone](weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        feat_dim = BACKBONE_FEATURES[backbone]
        self.proj = nn.Linear(feat_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
        return self.proj(f)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding over the token sequence."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len, 1, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        return self.dropout(x + self.pe[: x.size(0)])


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Boolean upper-triangular mask (True = attend is forbidden)."""
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )


class Im2VecModel(nn.Module):
    """Encoder-decoder that maps a raster image to an SVG token sequence."""

    def __init__(
        self,
        vocab_size: int = Vocab.VOCAB_SIZE,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        backbone: str = "resnet18",
        pretrained: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.encoder = ImageEncoder(d_model, backbone, pretrained)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
            images: (batch, 3, H, W) raster input.
            tokens: (batch, seq_len) token sequence (intended targets).
            padding_mask: (batch, seq_len) True where a token is PAD.

        Returns:
            logits: (batch, seq_len, vocab_size).
        """
        memory = self.encoder(images).unsqueeze(0)  # (1, batch, d_model)
        tgt = self.token_embed(tokens) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt.transpose(0, 1))  # (seq, batch, d_model)
        tgt_mask = _causal_mask(tgt.size(0), tgt.device)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)
        return self.head(out).transpose(0, 1)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_len: int,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Autoregressively sample a token sequence for each image.

        ``temperature == 0.0`` selects greedy decoding; otherwise tokens are
        sampled (optionally with nucleus ``top_p`` filtering).

        Returns:
            tokens: (batch, seq_len) including SOS/EOS.
        """
        self.eval()
        batch = images.size(0)
        device = images.device
        memory = self.encoder(images).unsqueeze(0)
        generated = torch.full((batch, 1), Vocab.SOS, dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt = self.token_embed(generated) * math.sqrt(self.d_model)
            tgt = self.pos_encoder(tgt.transpose(0, 1))
            tgt_mask = _causal_mask(tgt.size(0), device)
            out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.head(out[-1])  # (batch, vocab)

            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_p is not None:
                    logits = _top_p_filter(logits, top_p)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token], dim=1)
            if (generated[:, -1] == Vocab.EOS).all():
                break

        return generated


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out the tail of the distribution beyond cumulative probability ``p``."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum_probs > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    indices_to_remove = remove.scatter(1, sorted_idx, remove)
    return logits.masked_fill(indices_to_remove, float("-inf"))
