"""One-command FIGR-8 setup: download SVGs then render paired PNGs.

Usage::

    python -m im2vec.data.prepare --split train --n 5000 --out data/figr8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .download import download_figr_svg
from .render import render_svgs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download + render FIGR-8")
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

    n = download_figr_svg(args.out, split=args.split, n=args.n)
    print(f"downloaded {n} SVGs")

    if args.svg_only:
        return

    rendered, failed = render_svgs(args.out, size=args.size, workers=args.workers)
    print(f"rendered {rendered}, failed {failed}")
    print(f"dataset ready in {args.out.resolve()}")


if __name__ == "__main__":
    main()
