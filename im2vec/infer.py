"""Run a trained Im2Vec model on a raster image to produce an SVG."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .dataset import get_transform
from .model import Im2VecModel
from .tokenizer import SVGTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Im2Vec inference")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--method", choices=["greedy", "nucleus"], default="greedy")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-len", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
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
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    with Image.open(args.image) as im:
        x = get_transform()(im.convert("RGB")).unsqueeze(0).to(device)

    max_len = args.max_len or cfg.get("max_len", 256)
    temperature = 0.0 if args.method == "greedy" else args.temperature
    top_p = args.top_p if args.method == "nucleus" else None

    with torch.no_grad():
        tokens = model.generate(x, max_len, temperature=temperature, top_p=top_p)

    svg = tokenizer.decode_tokens(tokens[0].tolist())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
