"""Download SVG datasets from the Hugging Face Hub.

Supports any dataset stored in Parquet with an id column and an SVG-text
column (see :data:`DATASETS`). We resolve the Parquet shard URLs via the HF
API, download them, and extract one ``<id>.svg`` file per row.

Usage::

    python -m im2vec.data.download --dataset figr8 --split train --n 5000 --out data/svgs
    python -m im2vec.data.download --dataset svg-emoji --split train --out data/svgs

License: FIGR-8 is Creative Commons, non-commercial use only. svg-emoji is a
mix of Twemoji (CC-BY 4.0), Noto Emoji (Apache-2.0/OFL), and OpenMoji
(CC BY-SA 4.0, share-alike) — keep per-source attribution if redistributing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, TypedDict


class DatasetSpec(TypedDict):
    repo_id: str
    id_col: str
    svg_col: str
    splits: dict  # our split name -> the dataset's own split name
    license: str


DATASETS: dict = {
    "figr8": {
        "repo_id": "starvector/FIGR-SVG",
        "id_col": "Id",
        "svg_col": "Svg",
        "splits": {"train": "train", "valid": "valid", "test": "test"},
        "license": "CC BY-NC 4.0 (non-commercial use only) — monochrome pictograms",
    },
    "svg-emoji": {
        "repo_id": "starvector/svg-emoji",
        "id_col": "Filename",
        "svg_col": "Svg",
        "splits": {"train": "train", "valid": "val", "test": "test"},
        "license": (
            "mixed: Twemoji (CC-BY 4.0), Noto Emoji (Apache-2.0/OFL), "
            "OpenMoji (CC BY-SA 4.0, share-alike) — full-color icons"
        ),
    },
}


def _safe_stem(ident: str) -> str:
    """Sanitize an ``Id``/``Filename`` value into a filesystem-safe stem."""
    stem = Path(str(ident)).stem if str(ident).endswith(".svg") else str(ident)
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem)


def parquet_urls(dataset: str, split: str) -> List[str]:
    """Return the Parquet shard URLs for a dataset split."""
    spec = DATASETS[dataset]
    repo_id = spec["repo_id"]
    remote_split = spec["splits"].get(split, split)
    api = f"https://huggingface.co/api/datasets/{repo_id}/parquet/default"
    with urllib.request.urlopen(api, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if remote_split not in data:
        raise SystemExit(f"Unknown split {split!r} for {dataset!r}; available: {sorted(data)}")
    return data[remote_split]


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


def download_svgs(
    out_dir: Path,
    dataset: str = "figr8",
    split: str = "train",
    n: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> int:
    """Download SVGs for ``dataset``/``split`` into ``out_dir``; returns the count written."""
    import pyarrow.parquet as pq

    if dataset not in DATASETS:
        raise SystemExit(f"Unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")
    spec = DATASETS[dataset]
    id_col, svg_col = spec["id_col"], spec["svg_col"]

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or out_dir / ".parquet_cache"

    written = 0
    for url in parquet_urls(dataset, split):
        shard_name = url.rstrip("/").split("/")[-1]
        shard_path = cache_dir / dataset / split / shard_name
        _download(url, shard_path)

        pf = pq.ParquetFile(shard_path)
        for batch in pf.iter_batches(batch_size=1024):
            table = batch.to_pydict()
            ids = table[id_col]
            svgs = table[svg_col]
            for ident, svg in zip(ids, svgs):
                (out_dir / f"{_safe_stem(ident)}.svg").write_text(svg, encoding="utf-8")
                written += 1
                if n is not None and written >= n:
                    return written
    return written


def download_figr_svg(
    out_dir: Path,
    split: str = "train",
    n: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> int:
    """Backwards-compatible alias for ``download_svgs(dataset="figr8", ...)``."""
    return download_svgs(out_dir, dataset="figr8", split=split, n=n, cache_dir=cache_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download an SVG dataset")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="figr8")
    p.add_argument("--split", choices=["train", "valid", "test"], default="train")
    p.add_argument("--n", type=int, default=None, help="max SVGs to download")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = download_svgs(args.out, dataset=args.dataset, split=args.split, n=args.n)
    print(f"downloaded {n} SVGs to {args.out}  (license: {DATASETS[args.dataset]['license']})")


if __name__ == "__main__":
    main()
