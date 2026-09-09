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
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .inference import load_model, predict_svg

WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


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
