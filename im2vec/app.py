"""FastAPI web app: drag-and-drop raster-to-SVG converter.

Serves a small static frontend (``im2vec/web``) and a ``POST /api/convert``
endpoint that runs the trained model on an uploaded image and returns an SVG.

Run with::

    python -m im2vec.app --checkpoint checkpoints/model_epoch100.pt

or::

    uvicorn im2vec.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .dataset import get_transform
from .model import Im2VecModel
from .tokenizer import SVGTokenizer

WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def load_model(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> tuple[Im2VecModel, SVGTokenizer, dict, torch.device]:
    """Load a trained checkpoint into an ``Im2VecModel`` on the given device."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path} "
            f"(set IM2VEC_CHECKPOINT or pass --checkpoint)"
        )
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    tokenizer = SVGTokenizer()
    model = Im2VecModel(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.get("d_model", 512),
        nhead=cfg.get("nhead", 8),
        num_layers=cfg.get("num_layers", 6),
        dim_feedforward=cfg.get("dim_feedforward", 2048),
        max_len=cfg.get("max_len", 512),
        backbone=cfg.get("backbone", "resnet18"),
        pretrained=False,  # state_dict already contains the (fine-tuned) encoder
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, tokenizer, cfg, device


def predict_svg(
    model: Im2VecModel,
    tokenizer: SVGTokenizer,
    device: torch.device,
    image_bytes: bytes,
    max_len: int,
) -> str:
    """Run the model on raw image bytes and return the decoded SVG string."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = get_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = model.generate(x, max_len, temperature=0.0)
    return tokenizer.decode_tokens(tokens[0].tolist())


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = Path(
        os.environ.get("IM2VEC_CHECKPOINT", "checkpoints/model_epoch100.pt")
    )
    model, tokenizer, cfg, device = load_model(checkpoint)
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.config = cfg
    app.state.device = device
    print(f"loaded {checkpoint} on {device}")
    yield


app = FastAPI(title="Logo to SVG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/convert")
def convert(request: Request, file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, "Only PNG, JPEG, or WebP images are supported")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is too large (max 10 MB)")

    state = request.app.state
    max_len = state.config.get("max_len", 512)
    try:
        svg = predict_svg(
            state.model, state.tokenizer, state.device, data, max_len
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the client
        raise HTTPException(422, f"Could not process the image: {exc}") from exc

    return {"svg": svg, "width": 256, "height": 256}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Logo to SVG web app")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/model_epoch100.pt")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ["IM2VEC_CHECKPOINT"] = str(args.checkpoint)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
