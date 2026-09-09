"""Gradio app for the raster-to-SVG converter (Hugging Face Space, ZeroGPU).

Drag-and-drop a PNG/JPEG logo and download the predicted editable SVG. On a
ZeroGPU Space the ``convert`` function is wrapped with ``@GPU`` so inference
runs on the shared GPU; locally the decorator is a no-op and it falls back to
CPU.
"""

from __future__ import annotations

import io
import os
import tempfile
import urllib.parse
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

try:
    from spaces import GPU
except ImportError:  # pragma: no cover - local dev without the `spaces` package
    def GPU(fn=None, duration=None):
        def deco(f):
            return f

        if fn is not None:
            return deco(fn)
        return deco

from .inference import load_model, load_model_from_hub, predict_svg

_MODEL = None
_TOKENIZER = None
_CONFIG = None

CHECKER_BG = (
    "background-image:linear-gradient(45deg,#e5e7eb 25%,transparent 25%),"
    "linear-gradient(-45deg,#e5e7eb 25%,transparent 25%),"
    "linear-gradient(45deg,transparent 75%,#e5e7eb 75%),"
    "linear-gradient(-45deg,transparent 75%,#e5e7eb 75%);"
    "background-size:20px 20px;"
    "background-position:0 0,0 10px,10px -10px,-10px 0;"
)


def _ensure_loaded() -> None:
    """Load the model once (on CPU); called lazily and kept across requests."""
    global _MODEL, _TOKENIZER, _CONFIG
    if _MODEL is not None:
        return
    local = os.environ.get("IM2VEC_CHECKPOINT")
    if local:
        _MODEL, _TOKENIZER, _CONFIG, _ = load_model(
            Path(local), device=torch.device("cpu")
        )
    else:
        repo_id = os.environ.get("IM2VEC_MODEL_REPO", "R1l3y-w/im2vec-logo")
        filename = os.environ.get("IM2VEC_MODEL_FILE", "model_epoch100.pt")
        _MODEL, _TOKENIZER, _CONFIG, _ = load_model_from_hub(
            repo_id, filename, device=torch.device("cpu")
        )


def _render_preview(svg: str) -> str:
    """Embed the SVG in a checkered HTML preview (preserves transparency)."""
    encoded = urllib.parse.quote(svg)
    return (
        f"<div style='{CHECKER_BG} border-radius:8px; "
        "display:flex;align-items:center;justify-content:center;"
        "min-height:320px;padding:16px;'>"
        f"<img src='data:image/svg+xml;utf8,{encoded}' "
        "style='max-width:100%;max-height:420px;'/>"
        "</div>"
    )


@GPU
def convert(image: Image.Image | None) -> tuple[str, str | None]:
    """Run raster-to-SVG inference; returns (preview HTML, svg file path)."""
    if image is None:
        raise gr.Error("Please upload an image first.")

    _ensure_loaded()
    max_len = _CONFIG.get("max_len", 512)

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    data = buf.getvalue()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL.to(device)
    try:
        svg = predict_svg(_MODEL, _TOKENIZER, device, data, max_len)
    finally:
        _MODEL.to("cpu")

    tmpdir = tempfile.mkdtemp(prefix="logo-to-svg_")
    svg_path = os.path.join(tmpdir, "logo.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)

    return _render_preview(svg), svg_path


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Logo to SVG") as demo:
        gr.Markdown("# Logo to SVG")
        gr.Markdown(
            "Drop a PNG or JPEG logo to convert it into an editable SVG "
            "vector file. The first request loads the model and may take a "
            "few extra seconds."
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(
                    type="pil",
                    label="Upload logo",
                    image_mode="RGB",
                    sources=["upload"],
                )
                convert_btn = gr.Button("Convert", variant="primary")
            with gr.Column():
                preview = gr.HTML(label="Preview")
                download = gr.DownloadButton("Download .svg")

        convert_btn.click(convert, inputs=image, outputs=[preview, download])

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
