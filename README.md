# Im2Vec — Logo Raster-to-SVG

A PyTorch model that maps a raster logo (PNG/JPEG) to an editable SVG.

## Approach

The network is an **autoregressive encoder–decoder** that treats SVG as a
sequence of drawing commands:

- **Encoder** — a pretrained ResNet CNN extracts image features.
- **Decoder** — a Transformer decoder emits SVG tokens (commands, coordinates,
  and colors) one step at a time.

The SVG is normalized into a small vocabulary (`M`, `L`, `C`, `Z`, fill color)
and tokenized like text, so decoding produces a clean, editable `.svg`.

## Layout

```
im2vec/
  tokenizer.py   # SVG <-> token sequence (parse, normalize, quantize)
  model.py       # ResNet encoder + Transformer decoder
  dataset.py     # PyTorch Dataset over (raster, svg) pairs
  train.py       # teacher-forced training loop
  infer.py       # autoregressive sampling -> .svg
tests/
```

## Environment

```
conda activate pyt_env
pip install -r requirements.txt
```

## Dataset

Target dataset: **FIGR-8** logo/pictogram SVGs, rendered to PNG for paired
raster/vector supervision (see `README` planning notes). Non-commercial license.

## Status

Work in progress — neural-network core only (web tool to follow).
