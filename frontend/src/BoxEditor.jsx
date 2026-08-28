import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Stage, Layer, Image as KonvaImage, Rect, Transformer } from "react-konva";
import useImage from "use-image";
import { api } from "./api";

const DEFAULT_BOX = { w: 300, h: 200, angle: 0 };
const MIN_SCALE = 0.1;
const MAX_SCALE = 8;
const WHEEL_ZOOM_FACTOR = 1.06;

export default function BoxEditor({ batch, onSaved }) {
  const [method, setMethod] = useState(batch.method && batch.method !== "manual" ? batch.method : "background");
  const [meta, setMeta] = useState(null);
  const [boxes, setBoxes] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [dirty, setDirty] = useState(false);
  const [scale, setScale] = useState(1);

  const [image] = useImage(api.imageUrl(batch.batch_id));
  const shapeRefs = useRef({});
  const transformerRef = useRef(null);
  const containerRef = useRef(null);
  // Scroll position to apply after the next render (e.g. right after a
  // scale change resizes the scrollable content) — native scrollbars can't
  // be positioned until the element they scroll actually has its new size.
  const pendingScrollRef = useRef(null);
  // Synchronous mirror of `scale`, read/written by handleWheel. A physical
  // scroll wheel fires many events per gesture, often within one React
  // batch — computing each from the `scale` closure would have them all
  // read the same pre-batch value instead of compounding. setScale (async,
  // batched) drives rendering; this ref is the source of truth for math.
  const scaleRef = useRef(1);
  // The last box set the user actually saw as "current truth" — either just
  // loaded (detection ran, or a fresh batch) or just saved. Revert restores
  // this, undoing any edits made since without a round-trip to the server.
  const baselineBoxesRef = useRef([]);

  const setScaleBoth = (next) => {
    scaleRef.current = next;
    setScale(next);
  };

  // Fit the whole scan inside the current viewport; native scrollbars (not
  // this) take over for anything beyond that once zoomed in further. Reads
  // the container's size directly from the DOM rather than tracking it as
  // state — a ResizeObserver-driven size state here creates a feedback
  // loop, since zooming in can itself toggle the scrollbar and change the
  // measured size, which would silently re-trigger a fit and undo the zoom.
  const fitToView = () => {
    const el = containerRef.current;
    if (!el || !meta) return;
    const { width, height } = el.getBoundingClientRect();
    setScaleBoth(Math.min(width / meta.display_width, height / meta.display_height));
    pendingScrollRef.current = { left: 0, top: 0 };
  };

  // Re-fit when a new scan loads, and when the browser window itself is
  // resized (as opposed to the container resizing as a side effect of our
  // own zooming, which must NOT trigger a re-fit).
  useEffect(() => {
    fitToView();
    window.addEventListener("resize", fitToView);
    return () => window.removeEventListener("resize", fitToView);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta]);

  // Apply any scroll position queued by fitToView/handleWheel now that the
  // (possibly resized) scrollable content has actually rendered.
  useLayoutEffect(() => {
    if (pendingScrollRef.current && containerRef.current) {
      containerRef.current.scrollLeft = pendingScrollRef.current.left;
      containerRef.current.scrollTop = pendingScrollRef.current.top;
      pendingScrollRef.current = null;
    }
  }, [scale]);

  const handleWheel = (e) => {
    e.evt.preventDefault();
    const container = containerRef.current;
    if (!container || !meta) return;

    const currentScale = scaleRef.current;
    const rect = container.getBoundingClientRect();
    const pointerX = e.evt.clientX - rect.left;
    const pointerY = e.evt.clientY - rect.top;
    // Where the cursor is pointing, in unscaled scan-image coordinates —
    // this is what we keep fixed under the cursor as the zoom changes.
    const imageX = (container.scrollLeft + pointerX) / currentScale;
    const imageY = (container.scrollTop + pointerY) / currentScale;

    const direction = e.evt.deltaY > 0 ? -1 : 1;
    const raw = direction > 0 ? currentScale * WHEEL_ZOOM_FACTOR : currentScale / WHEEL_ZOOM_FACTOR;
    const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, raw));

    pendingScrollRef.current = {
      left: imageX * newScale - pointerX,
      top: imageY * newScale - pointerY,
    };
    setScaleBoth(newScale);
  };

  const loadBoxes = (m) => {
    setError(null);
    api
      .getBoxes(batch.batch_id, m)
      .then((resp) => {
        setBoxes(resp.boxes);
        baselineBoxesRef.current = resp.boxes.map((b) => ({ ...b }));
        setMeta(resp.meta);
        setSelectedIndex(null);
        setDirty(false);
      })
      .catch((e) => setError(e.message));
  };

  const revert = () => {
    setBoxes(baselineBoxesRef.current.map((b) => ({ ...b })));
    setSelectedIndex(null);
    setDirty(false);
  };

  useEffect(() => {
    loadBoxes(method);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [batch.batch_id]);

  useEffect(() => {
    if (selectedIndex != null && shapeRefs.current[selectedIndex] && transformerRef.current) {
      transformerRef.current.nodes([shapeRefs.current[selectedIndex]]);
      transformerRef.current.getLayer().batchDraw();
    } else if (transformerRef.current) {
      transformerRef.current.nodes([]);
    }
  }, [selectedIndex, boxes.length]);

  const updateBox = (index, patch) => {
    setBoxes((prev) => prev.map((b, i) => (i === index ? { ...b, ...patch } : b)));
    setDirty(true);
  };

  const addBox = () => {
    if (!meta) return;
    const newBox = {
      cx: meta.display_width / 2,
      cy: meta.display_height / 2,
      ...DEFAULT_BOX,
    };
    setBoxes((prev) => [...prev, newBox]);
    setSelectedIndex(boxes.length);
    setDirty(true);
  };

  const deleteSelected = () => {
    if (selectedIndex == null) return;
    setBoxes((prev) => prev.filter((_, i) => i !== selectedIndex));
    setSelectedIndex(null);
    setDirty(true);
  };

  const save = () => {
    if (!meta) return;
    setSaving(true);
    setError(null);
    api
      .saveCrop(batch.batch_id, boxes, meta.scale)
      .then((updatedBatch) => {
        setSaving(false);
        setDirty(false);
        baselineBoxesRef.current = boxes.map((b) => ({ ...b }));
        onSaved(updatedBatch);
      })
      .catch((e) => {
        setSaving(false);
        setError(e.message);
      });
  };

  return (
    <div className="box-editor">
      {batch.photos.length > 0 && (
        <div className="thumbnails-section">
          <h2>Images</h2>
          <div className="thumbnails">
            {batch.photos.map((p) => (
              <img key={p.index} src={api.photoUrl(p.jpeg_path, batch.timestamp)} alt={`photo ${p.index}`} />
            ))}
          </div>
        </div>
      )}
      <div className="box-editor-toolbar">
        <label>
          Detection method:{" "}
          <select
            value={method}
            onChange={(e) => {
              setMethod(e.target.value);
              loadBoxes(e.target.value);
            }}
          >
            <option value="background">background</option>
            <option value="edges">edges</option>
          </select>
        </label>
        <button onClick={addBox} disabled={!meta}>
          Add box
        </button>
        <button onClick={deleteSelected} disabled={selectedIndex == null}>
          Delete selected
        </button>
        <button onClick={fitToView} disabled={!meta}>
          Fit to screen
        </button>
        <span className="zoom-readout">{Math.round(scale * 100)}%</span>
        <span className="spacer" />
        {dirty && (
          <>
            <button onClick={revert} disabled={saving} className="link-button">
              Revert
            </button>
            <button onClick={save} disabled={saving || !meta} className="primary">
              {saving ? "Saving…" : batch.photos.length > 0 ? "Update photos" : "Save photos"}
            </button>
          </>
        )}
      </div>
      {error && <div className="error">{error}</div>}
      <div className="canvas-container" ref={containerRef}>
        {meta && (
          <Stage
            width={meta.display_width * scale}
            height={meta.display_height * scale}
            scaleX={scale}
            scaleY={scale}
            onWheel={handleWheel}
            onMouseDown={(e) => {
              if (e.target === e.target.getStage()) setSelectedIndex(null);
            }}
          >
            <Layer>
              {image && <KonvaImage image={image} width={meta.display_width} height={meta.display_height} />}
              {boxes.map((box, i) => (
                <Rect
                  key={i}
                  ref={(node) => (shapeRefs.current[i] = node)}
                  x={box.cx}
                  y={box.cy}
                  width={box.w}
                  height={box.h}
                  offsetX={box.w / 2}
                  offsetY={box.h / 2}
                  rotation={box.angle}
                  stroke={selectedIndex === i ? "#ff2d55" : "#39ff14"}
                  strokeWidth={3}
                  fill="rgba(57,255,20,0.15)"
                  draggable
                  onClick={() => setSelectedIndex(i)}
                  onTap={() => setSelectedIndex(i)}
                  onDragEnd={(e) => updateBox(i, { cx: e.target.x(), cy: e.target.y() })}
                  onTransformEnd={(e) => {
                    const node = e.target;
                    const newW = Math.max(10, node.width() * node.scaleX());
                    const newH = Math.max(10, node.height() * node.scaleY());
                    node.scaleX(1);
                    node.scaleY(1);
                    updateBox(i, {
                      cx: node.x(),
                      cy: node.y(),
                      w: newW,
                      h: newH,
                      angle: node.rotation(),
                    });
                  }}
                />
              ))}
              <Transformer ref={transformerRef} rotateEnabled anchorSize={22} anchorStrokeWidth={2} />
            </Layer>
          </Stage>
        )}
      </div>
    </div>
  );
}
