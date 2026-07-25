"""Detect, deskew, and crop individual photos out of a full-bed scan.

Assumes the photos were scanned against a black mat/background (for
contrast) and placed in a consistent reading orientation (so we only need
to correct skew, not figure out which way is "up" — that requires
understanding image content, not just geometry).
"""

from dataclasses import dataclass, asdict

import cv2
import numpy as np
from pathlib import Path

# Ignore contours smaller than this fraction of the full scan area (dust,
# mat seams, noise) or larger than this fraction (the whole bed, if the
# mat is missing/too bright).
MIN_AREA_FRACTION = 0.01
MAX_AREA_FRACTION = 0.95
# Real photos aren't extreme slivers; this catches background-seam artifacts
# (e.g. a mat that doesn't fully cover the bed, leaving a bright edge strip).
MAX_ASPECT_RATIO = 5.0
# Trim this many pixels inward after perspective correction, to remove any
# residual background sliver at the edges.
EDGE_MARGIN_PX = 6
JPEG_QUALITY = 95
# How many background-color standard deviations away (in Lab space) a pixel
# must be to count as "photo, not background". Distinguishing by color/
# texture distance from a sampled background reference (rather than plain
# brightness) is what lets a dark region *inside* a photo (hair, shadows,
# dark clothing) still be told apart from a background that also happens to
# be dark.
BACKGROUND_DISTANCE_THRESHOLD = 4.0
# Margin (as a fraction of the shorter image dimension) sampled from each
# edge of the scan to estimate the background color/texture. Photos should
# never be placed touching the very edge of the bed, so this strip is
# reliably background.
BACKGROUND_SAMPLE_MARGIN_FRACTION = 0.015

DETECTION_METHODS = ["background", "edges"]


@dataclass
class CroppedPhoto:
    index: int
    tiff_path: str
    jpeg_path: str
    width: int
    height: int


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left: smallest x+y
    ordered[2] = pts[np.argmax(s)]  # bottom-right: largest x+y
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # top-right: smallest y-x
    ordered[3] = pts[np.argmax(diff)]  # bottom-left: largest y-x
    return ordered


def _four_point_crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    corners = _order_corners(box.astype(np.float32))
    (tl, tr, br, bl) = corners

    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    width, height = max(width, 1), max(height, 1)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    m = EDGE_MARGIN_PX
    if width > 2 * m and height > 2 * m:
        warped = warped[m:height - m, m:width - m]
    return warped


def _background_reference(image: np.ndarray, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    m = max(4, int(min(h, w) * BACKGROUND_SAMPLE_MARGIN_FRACTION))
    border = np.concatenate([
        lab[:m, :].reshape(-1, 3),
        lab[-m:, :].reshape(-1, 3),
        lab[:, :m].reshape(-1, 3),
        lab[:, -m:].reshape(-1, 3),
    ])
    mean = border.mean(axis=0)
    # Floor the std so a near-perfectly-uniform background doesn't produce
    # an unreasonably tight (noise-sensitive) threshold.
    std = np.maximum(border.std(axis=0), 3.0)
    return mean, std


def _mask_background(image: np.ndarray) -> np.ndarray:
    """Classify pixels by color/texture distance from the sampled background.

    Good default: robust for typical photos, cheap, predictable. Weak spot:
    a region *inside* a photo that happens to closely match the background's
    specific color (not just "dark", but the same dark) is indistinguishable
    from background by color alone — that's an information limit, not a
    tuning problem. See `_mask_edges` for a method that doesn't rely on
    color at all.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_mean, bg_std = _background_reference(image, lab)

    z = np.abs(lab - bg_mean) / bg_std
    distance = np.linalg.norm(z, axis=2)
    mask = (distance > BACKGROUND_DISTANCE_THRESHOLD).astype(np.uint8) * 255

    # Scale the closing kernel to the scan's resolution so it can bridge
    # small internal gaps (specular highlights, print texture) that would
    # otherwise fragment a photo. A fixed small kernel doesn't scale to
    # multi-thousand-pixel 600dpi+ scans.
    close_size = max(15, min(image.shape[:2]) // 40)
    close_kernel = np.ones((close_size, close_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    open_kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return mask


def _mask_edges(image: np.ndarray) -> np.ndarray:
    """Trace the physical edge of each print instead of classifying color.

    A photo's cut edge is often detectable as a gradient (paper thickness
    shadow, a texture/gloss shift against the background) even when the
    photo's own content is the same color as the background right up to
    that edge — the case `_mask_background` can't handle. In practice this
    method actually succeeds two different ways: either the boundary itself
    traces as a closed loop, or (for a visually busy photo — foliage, a
    crowd, anything textured) the photo's *interior* is dense enough with
    edges that closing fuses it into one solid blob whose outer contour
    approximates the rectangle anyway. It reliably fails on a photo that is
    both same-colored as the background *and* visually smooth (e.g. a
    soft-focus portrait) — there's no boundary gradient and not enough
    interior texture either. That's a different failure mode than
    `_mask_background` (which fails on color, regardless of texture), not
    a strict superset of it — hence previewing before committing.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu picks a threshold that best separates two dominant intensity
    # modes, which stays sane even when one mode (the background) covers
    # most of the frame — a plain median degenerates to ~0 in that case
    # (over half the pixels are near-black background), flooding Canny
    # with noise instead of real edges.
    high, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = 0.5 * high
    edges = cv2.Canny(blurred, low, high)

    # Close (not dilate) small gaps in the traced boundary so broken
    # segments join into a loop `findContours` can trace. Closing bridges
    # gaps up to ~kernel-size without permanently growing the mask outward
    # everywhere the way a bare dilate does — a plain dilate here previously
    # inflated the mask from ~4% to ~34% coverage on a real scan, merging
    # unrelated regions into one useless blob. The gap being bridged is a
    # break in a traced line, not the space between separate photos, so
    # this doesn't need to scale with the scan's full resolution.
    close_size = int(np.clip(min(image.shape[:2]) / 400, 7, 21))
    close_kernel = np.ones((close_size, close_size), np.uint8)
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
    return mask


def _contours_from_mask(mask: np.ndarray, image_shape: tuple) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Photos are rectangular; take the convex hull to smooth over any
    # remaining notches from partial background misclassification, and
    # (for the edges method) to turn a hollow boundary ring into the solid
    # quadrilateral it outlines.
    contours = [cv2.convexHull(c) for c in contours]

    image_area = image_shape[0] * image_shape[1]
    min_area = image_area * MIN_AREA_FRACTION
    max_area = image_area * MAX_AREA_FRACTION

    kept = []
    for c in contours:
        # Use the same minAreaRect-derived box area for both the size and
        # aspect-ratio checks. Using raw hull area for one and the box for
        # the other lets a near-full-frame but non-rectangular hull (e.g.
        # the scanner bed's own edge) slip past the size filter while still
        # reading as ~square on the aspect check.
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        if min(w, h) == 0:
            continue
        box_area = w * h
        if not (min_area <= box_area <= max_area):
            continue
        if max(w, h) / min(w, h) > MAX_ASPECT_RATIO:
            continue
        kept.append(c)
    return kept


def find_photo_contours(image: np.ndarray, method: str = "background") -> list[np.ndarray]:
    if method == "background":
        mask = _mask_background(image)
    elif method == "edges":
        mask = _mask_edges(image)
    else:
        raise ValueError(f"Unknown detection method: {method!r} (expected one of {DETECTION_METHODS})")
    return _contours_from_mask(mask, image.shape)


def preview_detections(raw_scan_path: Path, method: str = "background", max_dim: int = 1200) -> np.ndarray:
    """Return a downscaled copy of the raw scan with detected boxes drawn on it."""
    image = cv2.imread(str(raw_scan_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image at {raw_scan_path}")

    contours = find_photo_contours(image, method=method)
    annotated = image.copy()
    thickness = max(4, min(image.shape[:2]) // 300)
    for c in contours:
        box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.intp)
        cv2.drawContours(annotated, [box], 0, (0, 255, 0), thickness=thickness)

    scale = max_dim / max(annotated.shape[:2])
    if scale < 1:
        annotated = cv2.resize(annotated, (int(annotated.shape[1] * scale), int(annotated.shape[0] * scale)))
    return annotated


def split_scan(raw_scan_path: Path, output_dir: Path, method: str = "background") -> list[CroppedPhoto]:
    image = cv2.imread(str(raw_scan_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image at {raw_scan_path}")

    contours = find_photo_contours(image, method=method)
    # Reading order: top-to-bottom, then left-to-right.
    contours.sort(key=lambda c: (cv2.boundingRect(c)[1] // 200, cv2.boundingRect(c)[0]))

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, contour in enumerate(contours, start=1):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        cropped = _four_point_crop(image, box)

        tiff_path = output_dir / f"photo_{i:02d}.tiff"
        jpeg_path = output_dir / f"photo_{i:02d}.jpg"
        cv2.imwrite(str(tiff_path), cropped)
        cv2.imwrite(str(jpeg_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        results.append(CroppedPhoto(
            index=i,
            tiff_path=str(tiff_path),
            jpeg_path=str(jpeg_path),
            width=cropped.shape[1],
            height=cropped.shape[0],
        ))
    return results


def photo_to_dict(photo: CroppedPhoto) -> dict:
    return asdict(photo)
