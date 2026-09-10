"""PyTorch dataset over (raster image, SVG) pairs."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .tokenizer import SVGTokenizer, Vocab

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def flatten_to_rgb(
    image: Image.Image, background: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """Composite a (possibly transparent) image onto ``background`` and drop alpha.

    ``Image.convert("RGB")`` does **not** alpha-composite — it just discards
    the alpha channel, exposing whatever RGB values sit underneath fully
    transparent pixels. Renderers such as cairosvg store ``(0, 0, 0)`` there,
    so a naive ``.convert("RGB")`` on a transparent-background PNG silently
    turns the entire background black. Always flatten through this function
    instead of calling ``.convert("RGB")`` directly on raster input.
    """
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        image = image.convert("RGBA")
        canvas = Image.new("RGB", image.size, background)
        canvas.paste(image, mask=image.split()[-1])
        return canvas
    return image.convert("RGB")


def load_manifest(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Pair up ``*.png``/``*.jpg``/``*.jpeg`` with same-stem ``*.svg`` files."""
    pairs: List[Tuple[Path, Path]] = []
    svgs = {p.stem: p for p in data_dir.glob("*.svg")}
    exts = ("*.png", "*.jpg", "*.jpeg")
    for ext in exts:
        for img in data_dir.glob(ext):
            svg = svgs.get(img.stem)
            if svg is not None:
                pairs.append((img, svg))
    return pairs


def filter_by_token_length(
    pairs: Sequence[Tuple[str, str]],
    tokenizer: SVGTokenizer,
    max_len: int,
) -> List[Tuple[Path, Path]]:
    """Drop ``(img, svg)`` pairs whose SVG tokenizes longer than ``max_len``.

    Silently-skips SVGs that fail to tokenize (rare, but keeps the run going).
    """
    kept: List[Tuple[Path, Path]] = []
    for img, svg in pairs:
        try:
            with open(svg, "r", encoding="utf-8") as f:
                n = len(tokenizer.encode_svg(f.read()))
        except Exception:
            continue
        if n <= max_len:
            kept.append((img, svg))
    return kept


def flip_tokens(tokens: Sequence[int], flip_x: bool, flip_y: bool) -> List[int]:
    """Mirror M/L/C coordinates (x/y over 0..255) in a token sequence.

    Color/width arguments (FILL/STROKE) are left untouched; only geometric
    coordinates are reflected, matching a raster ``FLIP_LEFT_RIGHT``/
    ``FLIP_TOP_BOTTOM`` of the corresponding 256x256 image.
    """
    out = list(tokens)
    max_val = Vocab.NUM_RANGE - 1  # 255
    i, n = 0, len(out)
    while i < n:
        tok = out[i]
        if tok in (Vocab.M, Vocab.L, Vocab.C):
            arity = Vocab.COMMAND_ARITY[tok]
            for j in range(i + 1, i + 1 + arity, 2):
                x = out[j] - Vocab.NUM_OFFSET
                y = out[j + 1] - Vocab.NUM_OFFSET
                if flip_x:
                    x = max_val - x
                if flip_y:
                    y = max_val - y
                out[j] = Vocab.NUM_OFFSET + x
                out[j + 1] = Vocab.NUM_OFFSET + y
            i += 1 + arity
        elif tok in (Vocab.FILL, Vocab.STROKE):
            i += 1 + Vocab.COMMAND_ARITY[tok]
        else:
            i += 1
    return out


def get_transform(image_size: int = 256) -> Callable:
    """Default raster transform matching the pretrained ResNet encoder."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class LogoDataset(Dataset):
    """Yields ``(image, token_sequence)`` pairs for a list of (img, svg) files.

    Args:
        pairs: list of ``(image_path, svg_path)`` tuples.
        tokenizer: an :class:`SVGTokenizer` instance.
        transform: raster transform (defaults to a 256x256 ImageNet-normalized
            transform if omitted).
        max_len: maximum token sequence length (SOS/EOS included).
    """

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        tokenizer: SVGTokenizer,
        transform: Optional[Callable] = None,
        max_len: int = 512,
        augment: bool = False,
    ):
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.transform = transform if transform is not None else get_transform()
        self.max_len = max_len
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, svg_path = self.pairs[idx]
        flip_x = flip_y = False
        with Image.open(img_path) as im:
            im = flatten_to_rgb(im)
            if self.augment:
                flip_x = random.random() < 0.5
                flip_y = random.random() < 0.5
                if flip_x:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                if flip_y:
                    im = im.transpose(Image.FLIP_TOP_BOTTOM)
            image = self.transform(im)

        with open(svg_path, "r", encoding="utf-8") as f:
            svg = f.read()
        tokens = self.tokenizer.encode_svg(svg)
        if self.augment and (flip_x or flip_y):
            tokens = flip_tokens(tokens, flip_x, flip_y)
        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len - 1] + [Vocab.EOS]
        return image, torch.tensor(tokens, dtype=torch.long)


def collate_fn(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length token sequences with PAD into a dense batch."""
    images = torch.stack([b[0] for b in batch])
    tokens = [b[1] for b in batch]
    max_len = max(t.size(0) for t in tokens)
    padded = torch.full((len(tokens), max_len), Vocab.PAD, dtype=torch.long)
    for i, t in enumerate(tokens):
        padded[i, : t.size(0)] = t
    return images, padded
