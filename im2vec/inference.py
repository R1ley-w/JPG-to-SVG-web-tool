"""Shared model loading and inference helpers.

Used by the FastAPI app (:mod:`im2vec.app`), the Gradio/ZeroGPU app
(:mod:`im2vec.gradio_app`), and the CLI (:mod:`im2vec.infer`).
"""

from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image

from .dataset import flatten_to_rgb, get_transform
from .model import Im2VecModel
from .tokenizer import SVGTokenizer


def load_model(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> tuple[Im2VecModel, SVGTokenizer, dict, torch.device]:
    """Load a trained checkpoint into an ``Im2VecModel`` on the given device."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    tokenizer = SVGTokenizer()
    model = Im2VecModel(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.get("d_model", 512),
        nhead=cfg.get("nhead", 8),
        num_layers=cfg.get("num_layers", 6),
        dim_feedforward=cfg.get("dim_feedforward", 2048),
        max_len=cfg.get("max_len", 512),
        backbone=cfg.get("backbone", "resnet18"),
        pretrained=False,  # state_dict already contains the (fine-tuned) encoder
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, tokenizer, cfg, device


def load_model_from_hub(
    repo_id: str,
    filename: str = "model_epoch100.pt",
    device: torch.device | None = None,
) -> tuple[Im2VecModel, SVGTokenizer, dict, torch.device]:
    """Download a checkpoint from the Hugging Face Hub and load it."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return load_model(Path(path), device=device)


def predict_svg(
    model: Im2VecModel,
    tokenizer: SVGTokenizer,
    device: torch.device,
    image_bytes: bytes,
    max_len: int,
) -> str:
    """Run the model on raw image bytes and return the decoded SVG string."""
    image = flatten_to_rgb(Image.open(io.BytesIO(image_bytes)))
    x = get_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = model.generate(x, max_len, temperature=0.0)
    return tokenizer.decode_tokens(tokens[0].tolist())
