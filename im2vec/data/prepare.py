"""One-command dataset setup: download SVGs then render paired PNGs.

Usage::

    python -m im2vec.data.prepare --dataset svg-emoji --split train --out data/svg-emoji/train
    python -m im2vec.data.prepare --dataset figr8 --split train --n 5000 --out data/figr8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .download import DATASETS, download_svgs
from .render import render_svgs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download + render an SVG dataset")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="figr8")
    p.add_argument("--split", choices=["train", "valid", "test"], default="train")
    p.add_argument("--n", type=int, default=None, help="max SVGs to download")
    p.add_argument("--out", type=Path, required=True, help="output dir for svg+png")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--svg-only", action="store_true", help="skip rendering")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    n = download_svgs(args.out, dataset=args.dataset, split=args.split, n=args.n)
    print(f"downloaded {n} SVGs  (license: {DATASETS[args.dataset]['license']})")

    if args.svg_only:
        return

    rendered, failed = render_svgs(args.out, size=args.size, workers=args.workers)
    print(f"rendered {rendered}, failed {failed}")
    print(f"dataset ready in {args.out.resolve()}")


if __name__ == "__main__":
    main()
