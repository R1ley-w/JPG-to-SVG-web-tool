"""PyTorch dataset over (raster image, SVG) pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .tokenizer import SVGTokenizer, Vocab

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
    ):
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.transform = transform if transform is not None else get_transform()
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, svg_path = self.pairs[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            image = self.transform(im)

        with open(svg_path, "r", encoding="utf-8") as f:
            svg = f.read()
        tokens = self.tokenizer.encode_svg(svg)
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
