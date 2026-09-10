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
app.py           # Hugging Face Space entry point (launches gradio_app.demo)
im2vec/
  tokenizer.py   # SVG <-> token sequence (parse, normalize, quantize)
  model.py       # ResNet encoder + Transformer decoder
  dataset.py     # PyTorch Dataset over (raster, svg) pairs
  train.py       # teacher-forced training loop
  infer.py       # autoregressive sampling -> .svg
  eval.py        # rasterize predictions; L1 / SSIM / IoU vs input
  data/
    download.py  # fetch SVG datasets from the HF Hub (parquet)
    render.py    # rasterize SVGs -> PNGs
    prepare.py   # one-command download + render
tests/
```

## Environment

```
conda activate pyt_env
pip install -r requirements.txt
```

## Dataset

Two SVG sources are supported out of the box (`im2vec/data/download.py:DATASETS`):

| `--dataset` | Source | Rows (train) | Color | License |
|---|---|---|---|---|
| `figr8` | `starvector/FIGR-SVG` | ~1.3M | **monochrome black only** (no `fill`/`style` in the source SVGs — see caveat below) | CC BY-NC 4.0 (non-commercial) |
| `svg-emoji` | `starvector/svg-emoji` | 8,708 | full color | mixed: Twemoji CC-BY 4.0, Noto Emoji Apache-2.0/OFL, OpenMoji CC BY-SA 4.0 (share-alike) |

> **FIGR-8 caveat:** every sampled FIGR-8 icon relies on the SVG default fill
> (black) — there is no color signal anywhere in that dataset, so a model
> trained only on it will never predict anything but black fills, regardless
> of the input image's colors. Use `svg-emoji` (or another colored source) to
> train a model that actually reproduces logo colors.

Download SVGs and render paired PNGs (one command):

```
python -m im2vec.data.prepare --dataset svg-emoji --split train --out data/svg-emoji/train
```

Or run the two steps separately:

```
python -m im2vec.data.download --dataset svg-emoji --split train --out data/svg-emoji/train
python -m im2vec.data.render  --svg-dir data/svg-emoji/train
```

`--split` is `train`/`valid`/`test`; `--n` caps the number of SVGs (omit for
all). PNGs are written alongside the SVGs (same stem) so
`train.py --data-dir data/svg-emoji/train` pairs them automatically. Repeat
for `--split valid` / `--split test` if you want held-out data for `eval.py`.

Recommended training flags for `svg-emoji` (long/complex icons, small
dataset): `--max-len 512 --augment`.

### Known fix: alpha backgrounds

Every raster load in this codebase goes through
`im2vec.dataset.flatten_to_rgb` rather than a bare `.convert("RGB")`. Plain
`Image.convert("RGB")` does **not** alpha-composite — it discards the alpha
channel and exposes whatever RGB values a renderer stored underneath fully
transparent pixels. cairosvg (and this repo's own `render.py`) stores
`(0, 0, 0)` there, so a naive `.convert("RGB")` silently turns every
transparent background solid black, both during training and in the Space's
`gradio_app.py` (and in Gradio's own `image_mode="RGB"` coercion, which has
the same bug — that's why `gradio_app.py` uses `image_mode=None` and flattens
itself). If you add a new image-loading code path, use `flatten_to_rgb`
instead of `.convert("RGB")`.

## Web app

Two frontends wrap the trained model. Both share `im2vec/inference.py`.

### Gradio (Hugging Face Space, ZeroGPU)

```
python -m im2vec.gradio_app
```

Drag a PNG/JPEG logo to convert it to an editable SVG. On a ZeroGPU Space the
inference runs on the shared GPU (`@spaces.GPU`); locally it falls back to CPU.

- Checkpoint is fetched from the Hub by default (`R1l3y-w/im2vec-logo`,
  `model_epoch100.pt`). Override with `IM2VEC_MODEL_REPO`/`IM2VEC_MODEL_FILE`,
  or point `IM2VEC_CHECKPOINT` at a local file.

### FastAPI (self-hosted)

```
python -m im2vec.app --checkpoint checkpoints/model_epoch100.pt
```

Then open <http://127.0.0.1:8000>. Drop a PNG/JPEG on the left; the SVG
preview and download button appear on the right.

- `--checkpoint` points to a trained checkpoint (default
  `checkpoints/model_epoch100.pt`); `IM2VEC_CHECKPOINT` is an alternative.
- `POST /api/convert` accepts an uploaded image and returns `{"svg": ...}`.

Run the tests (test deps: `pip install pytest httpx2`):

```
python -m pytest tests/ -q
```

## Status

Work in progress — neural-network core, data pipeline, and web apps are done.
