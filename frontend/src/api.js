const API_BASE = "http://localhost:8000";

async function json(resp) {
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || `${resp.status} ${resp.statusText}`);
  }
  return resp.json();
}

export const api = {
  listBatches: () => fetch(`${API_BASE}/api/batches`).then(json),
  startScan: () => fetch(`${API_BASE}/api/scan`, { method: "POST" }).then(json),
  imageUrl: (batchId) => `${API_BASE}/api/batches/${batchId}/image`,
  // `cacheBust` (e.g. the batch's timestamp) forces the browser to re-fetch
  // instead of showing a stale cached image — re-crops overwrite the same
  // filename (photo_01.jpg etc.) every time, so the URL never otherwise
  // changes even though the file content did.
  photoUrl: (jpegPath, cacheBust) => `${API_BASE}/${jpegPath}${cacheBust ? `?t=${encodeURIComponent(cacheBust)}` : ""}`,
  getBoxes: (batchId, method) =>
    fetch(`${API_BASE}/api/batches/${batchId}/boxes?method=${method}`).then(json),
  saveCrop: (batchId, boxes, scale) =>
    fetch(`${API_BASE}/api/batches/${batchId}/crop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ boxes, scale }),
    }).then(json),
};
