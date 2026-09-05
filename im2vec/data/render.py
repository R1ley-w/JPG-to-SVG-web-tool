"""Render downloaded FIGR-8 SVGs to paired PNG rasters.

Renders at ``size x size`` (default 256) to match the model's input transform
and the tokenizer's normalized canvas.

Usage::

    python -m im2vec.data.render --svg-dir data/svgs --png-dir data/pngs
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

DEFAULT_SIZE = 256


def render_one(
    svg_path: Path, png_path: Path, size: int = DEFAULT_SIZE
) -> Tuple[Path, bool, str]:
    """Render a single SVG; returns ``(path, ok, error)``."""
    try:
        import cairosvg

        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            output_width=size,
            output_height=size,
        )
        return svg_path, True, ""
    except Exception as exc:  # noqa: BLE001 - report any render failure
        return svg_path, False, str(exc)


def render_svgs(
    svg_dir: Path,
    png_dir: Path | None = None,
    size: int = DEFAULT_SIZE,
    workers: int | None = None,
    skip_existing: bool = True,
) -> Tuple[int, int]:
    """Render every ``*.svg`` in ``svg_dir`` into ``png_dir`` (defaults to the
    same directory, so PNGs and SVGs pair up for
    :func:`im2vec.dataset.load_manifest`).

    Returns ``(rendered, failed)``. PNGs keep the SVG stem.
    """
    png_dir = png_dir or svg_dir
    png_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, os.cpu_count() or 1)

    tasks: List[Tuple[Path, Path]] = []
    for svg in sorted(svg_dir.glob("*.svg")):
        png = png_dir / f"{svg.stem}.png"
        if skip_existing and png.exists():
            continue
        tasks.append((svg, png))

    rendered = failed = 0
    if not tasks:
        return rendered, failed

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(render_one, s, p, size) for s, p in tasks]
        for fut in as_completed(futures):
            path, ok, err = fut.result()
            if ok:
                rendered += 1
            else:
                failed += 1
                print(f"[fail] {path.name}: {err}", file=sys.stderr)

    return rendered, failed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render FIGR-8 SVGs to PNG")
    p.add_argument("--svg-dir", type=Path, required=True)
    p.add_argument("--png-dir", type=Path, default=None)
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rendered, failed = render_svgs(
        args.svg_dir, args.png_dir, size=args.size, workers=args.workers
    )
    print(f"rendered {rendered}, failed {failed}")


if __name__ == "__main__":
    main()
