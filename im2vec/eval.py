"""Evaluate a trained Im2Vec model: rasterize predicted SVGs and compare.

Metrics (computed against the input raster, which is the ground-truth render):
    * L1   — mean absolute error (0 = identical)
    * SSIM — structural similarity (1 = identical)
    * IoU  — foreground overlap of the binarized masks (1 = identical)
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import get_transform, load_manifest
from .model import Im2VecModel
from .tokenizer import SVGTokenizer

SIZE = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Im2Vec")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None, help="save predicted SVGs")
    p.add_argument("--n", type=int, default=None, help="limit number of samples")
    p.add_argument("--method", choices=["greedy", "nucleus"], default="greedy")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    return p.parse_args()


def _render_svg(svg: str) -> Image.Image:
    import cairosvg

    png = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"), output_width=SIZE, output_height=SIZE
    )
    return Image.open(io.BytesIO(png)).convert("RGBA")


def _rgb_over_white(img: Image.Image) -> np.ndarray:
    """Composite an RGBA image over white and return RGB floats in [0, 1]."""
    rgba = np.asarray(img.resize((SIZE, SIZE)), dtype=np.float32) / 255.0
    rgb = rgba[..., :3]
    alpha = rgba[..., 3:4]
    return rgb * alpha + (1.0 - alpha)


def _foreground(img: Image.Image) -> np.ndarray:
    alpha = np.asarray(img.resize((SIZE, SIZE)), dtype=np.float32)[..., 3]
    return alpha > 0.5


def _l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity (uniform 11x11 window) on RGB images in [0,1]."""
    ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    kernel = torch.ones(3, 1, 11, 11) / 121.0
    mu1 = F.conv2d(ta, kernel, padding=5, groups=3)
    mu2 = F.conv2d(tb, kernel, padding=5, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(ta * ta, kernel, padding=5, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(tb * tb, kernel, padding=5, groups=3) - mu2_sq
    sigma12 = F.conv2d(ta * tb, kernel, padding=5, groups=3) - mu1_mu2
    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim.mean())


def main() -> None:
    args = parse_args()
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
    model.eval()

    pairs = load_manifest(args.data_dir)
    if not pairs:
        raise SystemExit(f"No paired images/SVGs in {args.data_dir}")
    if args.n is not None:
        pairs = pairs[: args.n]

    temperature = 0.0 if args.method == "greedy" else args.temperature
    top_p = args.top_p if args.method == "nucleus" else None
    max_len = cfg.get("max_len", 256)

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    transform = get_transform(SIZE)
    l1s, ssims, ious = [], [], []

    for img_path, _ in pairs:
        with Image.open(img_path) as im:
            target = im.convert("RGBA")
            x = transform(im.convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            tokens = model.generate(x, max_len, temperature=temperature, top_p=top_p)
        svg = tokenizer.decode_tokens(tokens[0].tolist())

        if args.out_dir is not None:
            (args.out_dir / f"{img_path.stem}.svg").write_text(svg, encoding="utf-8")

        pred = _render_svg(svg)
        l1s.append(_l1(_rgb_over_white(pred), _rgb_over_white(target)))
        ssims.append(_ssim(_rgb_over_white(pred), _rgb_over_white(target)))
        ious.append(_iou(_foreground(pred), _foreground(target)))
        print(f"{img_path.name}: L1={l1s[-1]:.4f} SSIM={ssims[-1]:.4f} IoU={ious[-1]:.4f}")

    print(
        f"\nmean over {len(pairs)} samples -> "
        f"L1={np.mean(l1s):.4f}  SSIM={np.mean(ssims):.4f}  IoU={np.mean(ious):.4f}"
    )


if __name__ == "__main__":
    main()
