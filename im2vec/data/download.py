"""Download FIGR-8 SVG files from the Hugging Face Hub.

The `starvector/FIGR-SVG` dataset stores SVGs in Parquet (columns ``Id`` and
``Svg``). We resolve the Parquet shard URLs via the HF API, download them, and
extract one ``<Id>.svg`` file per row.

Usage::

    python -m im2vec.data.download --split train --n 5000 --out data/svgs

License: FIGR-8 is Creative Commons, non-commercial use only.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

API_DATASET = "starvector/FIGR-SVG"
PARQUET_API = f"https://huggingface.co/api/datasets/{API_DATASET}/parquet/default"


def parquet_urls(split: str) -> List[str]:
    """Return the Parquet shard URLs for a split (train/valid/test)."""
    with urllib.request.urlopen(PARQUET_API, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if split not in data:
        raise SystemExit(f"Unknown split {split!r}; available: {sorted(data)}")
    return data[split]


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)


def download_figr_svg(
    out_dir: Path,
    split: str = "train",
    n: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> int:
    """Download FIGR-8 SVGs into ``out_dir``; returns the number written."""
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or out_dir / ".parquet_cache"

    written = 0
    for url in parquet_urls(split):
        shard_name = url.rstrip("/").split("/")[-1]
        shard_path = cache_dir / split / shard_name
        _download(url, shard_path)

        pf = pq.ParquetFile(shard_path)
        for batch in pf.iter_batches(batch_size=1024):
            table = batch.to_pydict()
            ids = table["Id"]
            svgs = table["Svg"]
            for ident, svg in zip(ids, svgs):
                (out_dir / f"{ident}.svg").write_text(svg, encoding="utf-8")
                written += 1
                if n is not None and written >= n:
                    return written
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download FIGR-8 SVGs")
    p.add_argument("--split", choices=["train", "valid", "test"], default="train")
    p.add_argument("--n", type=int, default=None, help="max SVGs to download")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = download_figr_svg(args.out, split=args.split, n=args.n)
    print(f"downloaded {n} SVGs to {args.out}")


if __name__ == "__main__":
    main()
