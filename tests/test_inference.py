"""Tests for the shared inference helpers (im2vec.inference)."""

import io

import pytest
import torch
from PIL import Image

from im2vec.inference import load_model, predict_svg
from im2vec.model import Im2VecModel
from im2vec.tokenizer import SVGTokenizer


@pytest.fixture()
def tiny_checkpoint(tmp_path):
    """A tiny random-weight checkpoint (no network, no real training)."""
    tokenizer = SVGTokenizer()
    model = Im2VecModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        max_len=32,
        backbone="resnet18",
        pretrained=False,
    )
    cfg = {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 128,
        "max_len": 32,
        "backbone": "resnet18",
    }
    path = tmp_path / "tiny.pt"
    torch.save({"model": model.state_dict(), "config": cfg}, path)
    return path


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_load_and_predict(tiny_checkpoint):
    model, tokenizer, cfg, device = load_model(tiny_checkpoint)
    assert cfg["d_model"] == 64

    svg = predict_svg(model, tokenizer, device, _png_bytes(), cfg["max_len"])
    assert "<svg" in svg


def test_load_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does-not-exist.pt")
