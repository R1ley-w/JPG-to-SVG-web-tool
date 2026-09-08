# Im2Vec — Training Handoff

This document is a self-contained handoff for an agent (on another device) that
will **train** the raster-logo-to-SVG model. Everything below was built and
verified in the repo before this handoff; **no training has been run yet** on
real data.

---

## 1. What this is

A PyTorch model that maps a raster logo (PNG/JPEG) to an editable SVG.
Architecture: **CNN encoder (pretrained ResNet) + autoregressive Transformer
decoder** that emits SVG as a token sequence.

- Repo: `https://github.com/R1ley-w/JPG-to-SVG-web-tool.git` (branch `main`)
- Local dir on the original device: `…/Scripts&Tools/Im2Vec`
- Dataset: **FIGR-8** (logo/pictogram SVGs). **License: Creative Commons,
  non-commercial use only.**

---

## 2. Environment setup

```bash
git clone https://github.com/R1ley-w/JPG-to-SVG-web-tool.git
cd JPG-to-SVG-web-tool

# Recommended: create/use a Python 3.9+ env, then install a CUDA-matching torch.
# On the original device: conda env "pyt_env", Python 3.9.25, torch 2.6.0 (CPU).
pip install torch torchvision   # match CUDA on the training GPU
pip install -r requirements.txt
```

`requirements.txt` contents: `torch>=2.0`, `torchvision>=0.15`, `numpy`,
`Pillow`, `svgpathtools`, `cairosvg`, `pyarrow`.

- `cairosvg` needs native cairo. macOS: `brew install cairo libffi`.
  Linux: `apt-get install libcairo2-dev`.
- Code uses the **PyTorch 2.x API** (`torch.amp.GradScaler("cuda", …)`,
  `torch.load(..., weights_only=True)`). It auto-selects CUDA if available.
- Tests need `pytest` (run `pip install pytest`).

---

## 3. Repo layout

```
im2vec/
  tokenizer.py   # SVG <-> token sequence (parse, normalize, quantize)
  model.py       # ResNet encoder + Transformer decoder (+ sampling)
  dataset.py     # LogoDataset, collate_fn, get_transform, load_manifest
  train.py       # training loop (+ synthetic-data smoke test)
  infer.py       # checkpoint + PNG -> .svg
  eval.py        # L1 / SSIM / IoU vs input raster
  data/
    download.py  # fetch FIGR-8 SVGs from HF Hub (parquet)
    render.py    # rasterize SVGs -> 256x256 PNGs
    prepare.py   # one-command download + render
tests/
  test_tokenizer.py   # 7 tests (round-trip, shapes, arcs, strokes)
```

---

## 4. Tokenizer spec (important for the model)

SVG is normalized into a small vocabulary and encoded as an integer sequence.

Vocabulary (`im2vec/tokenizer.py`, `Vocab`), **vocab size = 265**:

| Token id | Meaning |
|----------|---------|
| 0  | `PAD` |
| 1  | `SOS` |
| 2  | `EOS` |
| 3  | `M` (move, 2 numbers) |
| 4  | `L` (line, 2 numbers) |
| 5  | `C` (cubic Bézier, 6 numbers) |
| 6  | `Z` (close, 0 numbers) |
| 7  | `FILL` (3 numbers: r,g,b) |
| 8  | `STROKE` (4 numbers: r,g,b,width) |
| 9..264 | numeric values `0..255` (token id = value + 9) |

Command arity map: `{M:2, L:2, C:6, Z:0, FILL:3, STROKE:4}`.

Encoded sequence format:

```
[SOS] ( FILL r g b <cmds> | STROKE r g b w <cmds> )* [EOS]
```

where `<cmds>` are `M x y | L x y | C x1 y1 x2 y2 x y | Z`.

- Coordinates are normalized to **0..255** against the SVG `viewBox`
  (non-uniform scale), matching a 256×256 canvas.
- All input paths are converted to absolute cubic Béziers: `H/V→L`,
  `S→C`, `Q/T→C`, `A(arc)→C`. Basic shapes (`rect/circle/ellipse/polygon/
  polyline`) are converted to paths. `transform` attributes (matrix/
  translate/scale/rotate) are applied.
- `decode_tokens()` produces an SVG with `viewBox="0 0 256 256"`.

---

## 5. Model spec

`im2vec/model.py` — `Im2VecModel`.

- **Encoder** `ImageEncoder`: ResNet backbone (default `resnet18`,
  ImageNet-pretrained) minus head → global avg pool → `Linear(feat_dim,
  d_model)`. (`resnet18/34` → 512 features; `resnet50` → 2048.)
- **Decoder**: `nn.TransformerDecoder` (default 6 layers, 8 heads,
  `d_model=512`, `dim_feedforward=2048`, `dropout=0.1`, `batch_first=False`),
  sinusoidal positional encoding, causal mask, cross-attention to the image
  embedding (a single memory token).
- **Head**: `Linear(d_model, vocab_size)`.
- `forward(images, tokens, padding_mask)` → teacher-forced logits
  `(batch, seq, vocab)`.
- `generate(images, max_len, temperature, top_p)` → autoregressive sampling
  (`temperature==0` = greedy).

Defaults live in `Im2VecModel.__init__`. A small smoke config (resnet18 +
2-layer/256-dim decoder) is ~13M params; the default 6-layer/512 config is
larger but still modest.

---

## 6. Data pipeline (FIGR-8)

One command downloads SVGs and renders paired PNGs:

```bash
python -m im2vec.data.prepare --split train --n 5000 --out data/figr8
```

Or separately:

```bash
python -m im2vec.data.download --split train --n 5000 --out data/figr8
python -m im2vec.data.render  --svg-dir data/figr8
```

- `--split`: `train` (1.3M rows) | `valid` (15k) | `test` (15k).
  Omit `--n` to download the whole split.
- Source: `starvector/FIGR-SVG` on Hugging Face (parquet, columns `Id`, `Svg`).
- PNGs are written **alongside** the SVGs (same stem, 256×256 RGBA) so
  `load_manifest` pairs them automatically. A `.parquet_cache/` subdir holds
  downloaded shards (safe to delete).

---

## 7. Training

```bash
python -m im2vec.train --data-dir data/figr8 --epochs 50 \
  --batch-size 16 --lr 1e-4 --max-len 512 --save-dir checkpoints
```

Key flags (see `train.py --help`):

| Flag | Default | Notes |
|------|---------|-------|
| `--data-dir` | — | dir of same-stem `*.png`/`*.jpg` + `*.svg` |
| `--epochs` | 20 | |
| `--batch-size` | 16 | tune to VRAM |
| `--lr` | 1e-4 | AdamW |
| `--d-model` | 512 | |
| `--nhead` | 8 | |
| `--num-layers` | 6 | |
| `--dim-ff` | 2048 | |
| `--backbone` | resnet18 | resnet18/34/50 |
| `--max-len` | 256 | **set to 512** (see caveat below) |
| `--save-dir` | checkpoints | |
| `--seed` | 0 | |
| `--smoke` | off | synthetic-data overfit test (no real data) |

Details:
- **Loss:** `CrossEntropyLoss` over tokens with `ignore_index=PAD`
  (teacher forcing: input `tokens[:, :-1]`, target `tokens[:, 1:]`).
- **Optimizer:** AdamW; mixed-precision AMP automatically on CUDA.
- **Checkpoints:** saved every 5 epochs as
  `{"model": state_dict, "config": dict}` (config is JSON-safe — no Path
  objects). `config` keys: `d_model, nhead, num_layers, dim_feedforward,
  max_len, backbone`.

Quick smoke test before real training (validates the whole pipeline on
synthetic shapes, no data needed):

```bash
python -m im2vec.train --smoke --smoke-n 16 --epochs 5 --batch-size 16 \
  --d-model 128 --nhead 4 --num-layers 2 --dim-ff 512 --max-len 64
```

Run the unit tests anytime:

```bash
python -m pytest tests/ -q
```

### ⚠️ max_len caveat (must fix before training)

Real FIGR-8 icons tokenize to **~300+ tokens** (one sampled icon was 328),
which exceeds `train.py`'s default `--max-len 256` and would be silently
truncated. **Use `--max-len 512`**, and consider adding a preprocessing
filter to drop out unusually long SVGs (recommended open item below).

---

## 8. Inference & evaluation

Inference (checkpoint + image → SVG):

```bash
python -m im2vec.infer --checkpoint checkpoints/model_epoch50.pt \
  --image logo.png --out logo.svg \
  [--method greedy|nucleus --temperature 1.0 --top-p 0.9]
```

Evaluation (rasterize predictions, compare to input raster):

```bash
python -m im2vec.eval --checkpoint checkpoints/model_epoch50.pt \
  --data-dir data/figr8 [--n 100 --out-dir out_svgs]
```

Metrics: **L1** (lower better), **SSIM** (higher better), **IoU** (higher
better) of the rendered prediction vs the input image.

---

## 9. Known limitations & decisions (do not "fix" without awareness)

- **Fill vs stroke:** if a shape has *both* `fill` and `stroke`, the tokenizer
  keeps the fill and drops the stroke. `fill="none"` shapes become `STROKE`
  paths. Gradients/`url()`/patterns fall back to black. `<text>` elements are
  not handled; `<line>` is skipped (no fillable area).
- **Stroke width** is scaled by the x-scale only (approximation for
  non-uniform viewBoxes).
- **Scale consistency:** rendering uses 256×256; tokenization maps to 0..255
  (a constant ~0.4% linear difference). This is consistent across all samples,
  so the model learns it — keep the same render size (256) everywhere.
- **Non-commercial license** (FIGR-8) — fine for research, not for a
  commercial product.

---

## 10. Recommended open items before/while training

1. **Length filter:** add a `--max-tokens` filter to the dataset/preprocessing
   to drop SVGs that tokenize above `max_len` (or set a sane `max_len=512`).
2. **Train/val split:** `load_manifest` currently pairs everything in a dir;
   wire up a proper train/valid split (use FIGR-8's `valid` split, or split
   the manifest).
3. **Data augmentation:** currently none beyond the resize/normalize transform.
   Add affine + color-jitter (apply the same affine to vector coords).
4. **Compute target:** single consumer GPU (≤24GB) was the stated target; the
   default config fits comfortably. Tune batch size for VRAM.

---

## 11. Session commit history (for reference)

```
2a12872 Add FIGR-8 data pipeline: download SVGs from HF and render paired PNGs
2dfca9a Document eval module in README
62e192e Add eval script (L1/SSIM/IoU); move load_manifest into dataset module
740c6ca Add stroke (outline) support to tokenizer vocab and encode/decode
37ca67f Add inference script; save clean JSON-safe config in checkpoints
023ccb2 Add dataset and training loop with synthetic-data smoke test
1e94173 Add Im2Vec model: ResNet encoder + Transformer decoder with sampling
0741c10 Add SVG tokenizer: parse/normalize SVG to M/L/C/Z token sequences
66f9cd6 Scaffold project: README, requirements, package layout
```
