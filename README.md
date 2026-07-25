# Auto Photo Scanner

A fully automated pipeline for bulk-digitizing family photos: scan several
prints at once on a flatbed, and have each one automatically detected,
deskewed, and cropped into its own file — no manual cropping or rotating.

Built and tested against a Brother DCP-J1200W on macOS. The scanning layer
is macOS-specific (see below); the detection/crop pipeline is plain
Python/OpenCV and should work with any source image.

## How it works

```
Streamlit UI --Start Scan--> scan.py (scanline/ImageCaptureCore)
                                  |
                                  v
                    data/raw/<timestamp>.tif  (full bed capture)
                                  |
                                  v
                       split.py (detect, deskew, crop)
                                  |
                    data/photos/<batch>/photo_NN.{tiff,jpg}
                                  |
                                  v
                    Streamlit UI --gallery + reprocess-->
```

## Requirements

- macOS (scanning goes through Apple's `ImageCaptureCore`)
- A scanner discoverable over AirScan/eSCL, or already set up as an ICA
  device
- Python 3
- [`scanline`](https://github.com/klep/scanline) installed and on `PATH`

## Setup

1. Install `scanline` — download the signed installer from its
   [releases](https://github.com/klep/scanline) (or build from source),
   run it, confirm `scanline -list` finds your scanner.
2. Grant **Local Network** permission to whatever terminal app you'll run
   this from (System Settings → Privacy & Security → Local Network).
   Scanner *discovery* works without this, but opening a scan session
   fails silently if it's missing.
3. Make sure no other app (Image Capture, Preview, etc.) has an open
   session with the scanner — `scanline` can't open its own session if
   one's already active, and the failure mode is a generic, unhelpful
   error.
4. Set up the Python environment:
   ```
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

## Running

```
.venv/bin/streamlit run app.py
```

Click **Start Scan** to scan the bed and split it into individual photos.
Past batches are listed below with a gallery of their photos.

## Physical scanning setup

- **Use a dark, uniform background.** Lay a black mat/board/cloth over the
  photos before closing the lid — the scanner's own lid is white, and
  photo-to-background contrast is what detection relies on. Make sure it
  covers the *entire* bed; a background that doesn't reach the edges
  leaves a bright seam that can get detected as a spurious extra "photo".
- **Place every photo in the same reading orientation.** The pipeline only
  auto-corrects *skew* (a crooked angle) — it does not try to detect or
  fix upside-down or sideways placement, since that requires understanding
  the photo's content, not just its geometry. This is a deliberate
  tradeoff to keep the software side fully automatic; consistent placement
  on your end is what makes that possible.

## Known limitations

- **Resolution is capped at 600dpi**, well under this scanner's 1200x2400
  optical hardware maximum. Brother doesn't ship a real macOS ICA driver
  for this consumer model — only Apple's built-in AirScan (eSCL) protocol
  works, and eSCL caps this specific device at 600dpi with JPEG/PDF output
  only. We still request TIFF output from `scanline` to avoid a second
  lossy re-compression pass on our end, but the source data was already
  JPEG-compressed by the scanner itself — this isn't true archival-original
  quality, just the best this device can hand over.
- **Two detection methods, two different blind spots** — neither is a
  strict improvement on the other:
  - `background` (default): classifies pixels by color/texture distance
    from the sampled background. Fails when part of a photo happens to
    match the background's specific color (e.g. a dark jacket against a
    dark background), regardless of how detailed that region is.
  - `edges`: traces each photo's physical edge via Canny. Fails when a
    photo is *both* same-colored as the background *and* visually smooth
    (e.g. a soft-focus portrait) — there's not enough gradient signal
    anywhere to form a closed boundary.
  - Use the **"Reprocess from the original scan"** panel on any past batch
    to preview detection with either method and re-split without
    rescanning.
- **No automatic orientation correction** — see "Physical scanning setup"
  above.

## Project layout

```
scan_pipeline/
  scan.py       # drives `scanline` to capture the full bed
  split.py      # detect, deskew, and crop individual photos
  pipeline.py   # orchestration + batch metadata
app.py          # Streamlit UI
data/           # scans (gitignored): raw/ + photos/<batch_id>/
```
