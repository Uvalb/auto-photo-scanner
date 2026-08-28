"""Thin HTTP wrapper around scan_pipeline for the React frontend.

Run with: .venv/bin/uvicorn backend.api:app --reload --port 8000
"""

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scan_pipeline import pipeline, scan, split

app = FastAPI(title="Photo Scanner API")

# The frontend runs on Vite's dev server (a different origin) during
# development; the backend and frontend are separate processes. Vite picks
# the next free port if 5173 is taken, so allow any localhost dev port
# rather than hardcoding one.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve cropped photo thumbnails directly (data/photos/<batch>/photo_NN.jpg).
# The raw scans are not served this way — they're huge TIFFs a browser
# can't decode, so those go through /batches/{id}/image instead.
app.mount("/data", StaticFiles(directory="data"), name="data")


class Box(BaseModel):
    cx: float
    cy: float
    w: float
    h: float
    angle: float


class CropRequest(BaseModel):
    boxes: List[Box]
    scale: float


@app.get("/api/batches")
def list_batches():
    return pipeline.list_batches()


@app.post("/api/scan")
def start_scan():
    try:
        return pipeline.run_batch()
    except scan.ScanError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/batches/{batch_id}/image")
def batch_image(batch_id: str):
    try:
        image_bytes = pipeline.get_display_image(batch_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/api/batches/{batch_id}/boxes")
def batch_boxes(batch_id: str, method: str = "background"):
    if method not in split.DETECTION_METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown method {method!r} (expected one of {split.DETECTION_METHODS})")
    try:
        boxes, meta = pipeline.get_display_boxes(batch_id, method)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"boxes": boxes, "meta": meta}


@app.post("/api/batches/{batch_id}/crop")
def crop_batch(batch_id: str, req: CropRequest):
    try:
        return pipeline.save_boxes(batch_id, [b.model_dump() for b in req.boxes], req.scale)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
