import { useEffect, useState } from "react";
import { api } from "./api";
import BoxEditor from "./BoxEditor";
import "./App.css";

export default function App() {
  const [batches, setBatches] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState(null);

  const refreshBatches = () => api.listBatches().then(setBatches).catch((e) => setError(e.message));

  useEffect(() => {
    refreshBatches();
  }, []);

  const startScan = () => {
    setScanning(true);
    setError(null);
    api
      .startScan()
      .then((batch) => {
        setScanning(false);
        setSelectedId(batch.batch_id);
        refreshBatches();
      })
      .catch((e) => {
        setScanning(false);
        setError(e.message);
      });
  };

  const selectedBatch = batches.find((b) => b.batch_id === selectedId);

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Photo Scanner</h1>
        <button onClick={startScan} disabled={scanning} className="primary">
          {scanning ? "Scanning…" : "Start Scan"}
        </button>
        {error && <div className="error">{error}</div>}
        <h2>Batches</h2>
        <ul className="batch-list">
          {batches.map((b) => (
            <li
              key={b.batch_id}
              className={b.batch_id === selectedId ? "selected" : ""}
              onClick={() => setSelectedId(b.batch_id)}
            >
              <div>{b.timestamp}</div>
              <div className="batch-meta">
                {b.photo_count} photo(s) — {b.method || "background"}
              </div>
            </li>
          ))}
        </ul>
      </aside>
      <main className="main">
        {!selectedBatch && <p>Select a batch, or start a new scan.</p>}
        {selectedBatch && (
          <BoxEditor
            key={selectedBatch.batch_id}
            batch={selectedBatch}
            onSaved={() => refreshBatches()}
          />
        )}
      </main>
    </div>
  );
}
