"""Training loop for the Im2Vec raster-to-SVG model."""

from __future__ import annotations

import argparse
import random
import re
import tempfile
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import (
    LogoDataset,
    collate_fn,
    filter_by_token_length,
    get_transform,
    load_manifest,
)
from .model import Im2VecModel
from .tokenizer import SVGTokenizer, Vocab


def make_synthetic_data(
    n: int, out_dir: Path, image_size: int = 256
) -> List[Tuple[Path, Path]]:
    """Generate simple random shape SVGs + rendered PNGs (smoke-test data)."""
    import cairosvg

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs: List[Tuple[Path, Path]] = []
    shapes = ["circle", "rect", "ellipse", "polygon"]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f1c40f", "#9b59b6", "#1abc9c"]

    for i in range(n):
        kind = random.choice(shapes)
        fill = random.choice(colors)
        if kind == "circle":
            cx, cy, r = random.randint(40, 216), random.randint(40, 216), random.randint(20, 70)
            body = f'<circle cx="{cx}" cy="{cy}" r="{r}"/>'
        elif kind == "rect":
            x, y = random.randint(20, 150), random.randint(20, 150)
            w, h = random.randint(30, 120), random.randint(30, 120)
            body = f'<rect x="{x}" y="{y}" width="{w}" height="{h}"/>'
        elif kind == "ellipse":
            cx, cy = random.randint(40, 216), random.randint(40, 216)
            rx, ry = random.randint(20, 70), random.randint(20, 70)
            body = f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}"/>'
        else:
            x, y = random.randint(30, 150), random.randint(30, 150)
            s = random.randint(40, 100)
            body = (
                f'<polygon points="{x},{y} {x+s},{y} {x+s},{y+s} '
                f'{x},{y+s}"/>'
            )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {image_size} '
            f'{image_size}"><g fill="{fill}">{body}</g></svg>'
        )
        svg_path = out_dir / f"{i:04d}.svg"
        png_path = out_dir / f"{i:04d}.png"
        svg_path.write_text(svg, encoding="utf-8")
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(png_path),
            output_width=image_size,
            output_height=image_size,
        )
        pairs.append((png_path, svg_path))
    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Im2Vec")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        nargs="+",
        help="one or more dirs of paired PNG/JPG + SVG files (combined into one training set)",
    )
    p.add_argument("--smoke", action="store_true", help="overfit synthetic data (pipeline test)")
    p.add_argument("--smoke-n", type=int, default=16, help="synthetic samples for --smoke")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dim-ff", type=int, default=2048)
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--augment", action="store_true", help="random flips (raster + vector)")
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume from a checkpoint (epoch inferred from filename)",
    )
    return p.parse_args()


def _epoch_from_path(p: Path) -> int:
    """Infer the epoch number from a checkpoint filename (``model_epoch50.pt``)."""
    m = re.search(r"epoch(\d+)", p.name)
    return int(m.group(1)) if m else 0


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    tokenizer = SVGTokenizer()

    state_dict = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        state_dict = ckpt["model"]
        cfg = ckpt.get("config", {})
        model_cfg = {
            "d_model": cfg.get("d_model", args.d_model),
            "nhead": cfg.get("nhead", args.nhead),
            "num_layers": cfg.get("num_layers", args.num_layers),
            "dim_feedforward": cfg.get("dim_feedforward", args.dim_ff),
            "max_len": cfg.get("max_len", args.max_len),
            "backbone": cfg.get("backbone", args.backbone),
        }
        start_epoch = _epoch_from_path(args.resume)
        print(f"resuming from {args.resume} (epoch {start_epoch})")
    else:
        model_cfg = {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_ff,
            "max_len": args.max_len,
            "backbone": args.backbone,
        }
        start_epoch = 0

    max_len = model_cfg["max_len"]

    if args.smoke:
        tmp = Path(tempfile.mkdtemp(prefix="im2vec_smoke_"))
        pairs = make_synthetic_data(args.smoke_n, tmp)
        data_dir = tmp
    elif args.data_dir is not None:
        pairs = []
        for d in args.data_dir:
            found = load_manifest(d)
            print(f"{len(found)} samples in {d}")
            pairs.extend(found)
        data_dir = args.data_dir[0]
    else:
        raise SystemExit("Provide --data-dir or use --smoke")

    if not pairs:
        raise SystemExit(f"No paired images/SVGs found in {data_dir}")
    print(f"{len(pairs)} samples (before length filter)")

    pairs = filter_by_token_length(pairs, tokenizer, max_len)
    if not pairs:
        raise SystemExit(f"No samples left after length filter (max_len={max_len})")
    print(f"{len(pairs)} samples after length filter (max_len={max_len})")

    dataset = LogoDataset(pairs, tokenizer, max_len=max_len, augment=args.augment)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    model = Im2VecModel(
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        max_len=max_len,
        backbone=model_cfg["backbone"],
    ).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)

    config = model_cfg

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    total_epochs = start_epoch + args.epochs
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total = 0.0
        for images, tokens in loader:
            images = images.to(device)
            tokens = tokens.to(device)
            inp = tokens[:, :-1]
            tgt = tokens[:, 1:]
            pad_mask = inp == Vocab.PAD

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images, inp, padding_mask=pad_mask)
                loss = F.cross_entropy(
                    logits.reshape(-1, tokenizer.vocab_size),
                    tgt.reshape(-1),
                    ignore_index=Vocab.PAD,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()

        avg = total / max(1, len(loader))
        print(f"epoch {epoch + 1:3d}/{total_epochs}  loss={avg:.4f}", flush=True)

        if (epoch + 1) % 5 == 0 or args.smoke:
            torch.save(
                {"model": model.state_dict(), "config": config},
                args.save_dir / f"model_epoch{epoch + 1}.pt",
            )

    print(f"checkpoints in {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()
