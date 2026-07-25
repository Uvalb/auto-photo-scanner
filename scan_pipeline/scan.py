"""Scan acquisition via the `scanline` CLI (drives macOS ImageCaptureCore).

For the Brother DCP-J1200W, scanning goes through Apple's built-in
AirScanScanner module over eSCL/AirScan — Brother doesn't ship a real
macOS ICA driver for this consumer model, only a USB "push scan" helper
that's irrelevant here. eSCL caps this device at 600dpi and JPEG/PDF only,
well under its 1200x2400 optical hardware max; see MAX_RESOLUTION_DPI.
Requesting "-tiff" still avoids a second lossy re-compression pass on our
end, even though the source data was already JPEG-compressed by the
scanner itself — it's not true archival-original quality, just the best
this device can hand over.

Requires:
  - `scanline` installed and on PATH (https://github.com/klep/scanline).
  - macOS's "Local Network" privacy permission granted to whatever
    terminal/process runs this — discovery works without it, but opening
    a scan session silently fails.
  - No other app (e.g. Image Capture) holding an open session with the
    scanner at the same time.
"""

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

MAX_RESOLUTION_DPI = 600


class ScanError(RuntimeError):
    pass


def _scanline_bin() -> str:
    path = shutil.which("scanline")
    if path is None:
        raise ScanError(
            "scanline not found on PATH. Install it from "
            "https://github.com/klep/scanline and ensure it's on PATH."
        )
    return path


def scan_full_bed(raw_dir: Path, resolution: int = MAX_RESOLUTION_DPI) -> Path:
    """Scan the full flatbed at the given resolution, saved as a lossless TIFF.

    Returns the path to the saved raw scan.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S")

    cmd = [
        _scanline_bin(),
        "-flatbed",
        "-tiff",
        "-res", str(resolution),
        "-dir", str(raw_dir),
        "-name", name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise ScanError(f"scanline failed: {result.stdout}\n{result.stderr}")

    scan_path = raw_dir / f"{name}.tif"
    if not scan_path.exists():
        raise ScanError(
            f"Expected scan output at {scan_path} but it wasn't created. "
            f"scanline output:\n{result.stdout}\n{result.stderr}"
        )
    return scan_path
