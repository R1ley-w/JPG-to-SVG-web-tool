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
  eval.py        # rasterize predictions; L1 / SSIM / IoU vs input
tests/
```

## Environment

```
conda activate pyt_env
pip install -r requirements.txt
```

## Dataset

Target dataset: **FIGR-8** logo/pictogram SVGs, rendered to PNG for paired
raster/vector supervision. Non-commercial license.

Download SVGs and render paired PNGs (one command):

```
python -m im2vec.data.prepare --split train --n 5000 --out data/figr8
```

Or run the two steps separately:

```
python -m im2vec.data.download --split train --n 5000 --out data/figr8
python -m im2vec.data.render  --svg-dir data/figr8
```

`--split` is `train`/`valid`/`test`; `--n` caps the number of SVGs (omit for
all). PNGs are written alongside the SVGs (same stem) so
`train.py --data-dir data/figr8` pairs them automatically.

## Web app

Run the drag-and-drop converter:

```
python -m im2vec.app --checkpoint checkpoints/model_epoch100.pt
```

Then open <http://127.0.0.1:8000>. Drop a PNG/JPEG on the left; the SVG
preview and download button appear on the right.

- The model loads once at startup (CUDA if available, else CPU).
- `--checkpoint` points to a trained checkpoint (default
  `checkpoints/model_epoch100.pt`); `IM2VEC_CHECKPOINT` is an alternative.
- `POST /api/convert` accepts an uploaded image and returns `{"svg": ...}`.

Run the tests (test deps: `pip install pytest httpx2`):

```
python -m pytest tests/ -q
```

## Status

Work in progress — neural-network core, data pipeline, and web tool are done.
