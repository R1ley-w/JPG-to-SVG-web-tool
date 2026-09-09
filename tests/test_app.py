"""Tests for the FastAPI web app (im2vec.app)."""

import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from im2vec.model import Im2VecModel
from im2vec.tokenizer import SVGTokenizer


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Build a tiny random-weight model and serve it through TestClient."""
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
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"model": model.state_dict(), "config": cfg}, checkpoint)
    monkeypatch.setenv("IM2VEC_CHECKPOINT", str(checkpoint))

    from im2vec import app as app_module

    with TestClient(app_module.app) as c:
        yield c


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_index_serves_frontend(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Logo to SVG" in resp.text


def test_convert_returns_svg(client):
    resp = client.post(
        "/api/convert", files={"file": ("logo.png", _png_bytes(), "image/png")}
    )
    assert resp.status_code == 200
    assert "<svg" in resp.json()["svg"]


def test_convert_rejects_wrong_type(client):
    resp = client.post(
        "/api/convert", files={"file": ("note.txt", b"hello", "text/plain")}
    )
    assert resp.status_code == 415


def test_convert_rejects_corrupt_image(client):
    resp = client.post(
        "/api/convert", files={"file": ("bad.png", b"not-an-image", "image/png")}
    )
    assert resp.status_code == 422
