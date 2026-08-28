"""Glue: scan the bed, split it into individual photos, record batch metadata."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import scan, split

RAW_DIR = Path("data/raw")
PHOTOS_DIR = Path("data/photos")


def run_batch(resolution: int = scan.MAX_RESOLUTION_DPI, method: str = "background") -> dict:
    raw_path = scan.scan_full_bed(RAW_DIR, resolution=resolution)
    batch_id = raw_path.stem
    batch_dir = PHOTOS_DIR / batch_id

    photos = split.split_scan(raw_path, batch_dir, method=method)

    metadata = {
        "batch_id": batch_id,
        "timestamp": datetime.now().isoformat(),
        "raw_scan": str(raw_path),
        "resolution_dpi": resolution,
        "method": method,
        "photo_count": len(photos),
        "photos": [split.photo_to_dict(p) for p in photos],
    }
    (batch_dir / "batch.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def raw_scan_path(batch_id: str) -> Path:
    return RAW_DIR / f"{batch_id}.tif"


def preview_batch(batch_id: str, method: str):
    """Return an annotated preview image (no files written) for a past batch's raw scan."""
    return split.preview_detections(raw_scan_path(batch_id), method=method)


def _replace_batch_photos(batch_id: str, photos: list, method_label: str, extra: Optional[dict] = None) -> dict:
    """Overwrite a batch's cropped photos + metadata in place (same batch_id,
    same underlying raw scan) — shared by reprocess_batch and save_boxes."""
    batch_dir = PHOTOS_DIR / batch_id
    existing_json = batch_dir / "batch.json"
    metadata = json.loads(existing_json.read_text()) if existing_json.exists() else {}

    metadata.update({
        "batch_id": batch_id,
        "timestamp": datetime.now().isoformat(),
        "raw_scan": str(raw_scan_path(batch_id)),
        "method": method_label,
        "photo_count": len(photos),
        "photos": [split.photo_to_dict(p) for p in photos],
        **(extra or {}),
    })
    existing_json.write_text(json.dumps(metadata, indent=2))
    return metadata


def reprocess_batch(batch_id: str, method: str) -> dict:
    """Re-run detection/crop on an already-scanned raw image with a different method.

    Overwrites the batch's existing cropped photos in place (same batch_id)
    rather than creating a new batch, since it's the same underlying scan.
    """
    raw_path = raw_scan_path(batch_id)
    if not raw_path.exists():
        raise FileNotFoundError(f"No raw scan found for batch {batch_id!r} at {raw_path}")
    batch_dir = PHOTOS_DIR / batch_id

    for stale in batch_dir.glob("photo_*.tiff"):
        stale.unlink()
    for stale in batch_dir.glob("photo_*.jpg"):
        stale.unlink()

    photos = split.split_scan(raw_path, batch_dir, method=method)
    return _replace_batch_photos(batch_id, photos, method)


def get_display_image(batch_id: str) -> bytes:
    return split.display_image_bytes(raw_scan_path(batch_id))


def get_display_boxes(batch_id: str, method: str) -> tuple[list[dict], dict]:
    return split.find_display_boxes(raw_scan_path(batch_id), method=method)


def save_boxes(batch_id: str, boxes: list[dict], scale: float) -> dict:
    """Crop from manually-edited boxes (in a `scale`d-down coordinate space)
    and overwrite the batch's photos in place."""
    raw_path = raw_scan_path(batch_id)
    if not raw_path.exists():
        raise FileNotFoundError(f"No raw scan found for batch {batch_id!r} at {raw_path}")
    batch_dir = PHOTOS_DIR / batch_id

    for stale in batch_dir.glob("photo_*.tiff"):
        stale.unlink()
    for stale in batch_dir.glob("photo_*.jpg"):
        stale.unlink()

    photos = split.crop_from_boxes(raw_path, batch_dir, boxes, scale=scale)
    return _replace_batch_photos(batch_id, photos, "manual", extra={"boxes": boxes})


def list_batches() -> list[dict]:
    if not PHOTOS_DIR.exists():
        return []
    batches = []
    for batch_json in PHOTOS_DIR.glob("*/batch.json"):
        try:
            batches.append(json.loads(batch_json.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    batches.sort(key=lambda b: b["timestamp"], reverse=True)
    return batches
