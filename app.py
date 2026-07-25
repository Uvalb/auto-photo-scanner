import streamlit as st

from scan_pipeline import pipeline, split

st.set_page_config(page_title="Photo Scanner", layout="wide")
st.title("Family Photo Scanner")

if st.button("Start Scan", type="primary"):
    with st.spinner("Scanning at full resolution and splitting photos..."):
        try:
            batch = pipeline.run_batch()
        except Exception as e:
            st.error(str(e))
        else:
            st.success(f"Scanned {batch['photo_count']} photo(s).")

st.divider()
st.subheader("Results")

batches = pipeline.list_batches()
if not batches:
    st.info("No scans yet. Click \"Start Scan\" to begin.")

for batch in batches:
    method_used = batch.get("method", "background")
    st.markdown(f"**{batch['timestamp']}** — {batch['photo_count']} photo(s) — detected with `{method_used}`")
    cols = st.columns(min(len(batch["photos"]), 6) or 1)
    for i, photo in enumerate(batch["photos"]):
        with cols[i % len(cols)]:
            st.image(photo["jpeg_path"], use_container_width=True)

    with st.expander("Reprocess from the original scan"):
        st.caption(
            "Didn't detect the photos correctly? Re-run detection on the same raw "
            "scan with a different method — no need to rescan."
        )
        batch_id = batch["batch_id"]
        method = st.selectbox(
            "Detection method",
            split.DETECTION_METHODS,
            index=split.DETECTION_METHODS.index(method_used) if method_used in split.DETECTION_METHODS else 0,
            key=f"method_{batch_id}",
        )
        preview_key = f"preview_{batch_id}"
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Preview detection", key=f"preview_btn_{batch_id}"):
                with st.spinner("Running detection..."):
                    st.session_state[preview_key] = pipeline.preview_batch(batch_id, method)
        with col_b:
            if st.button("Apply and re-split", key=f"apply_btn_{batch_id}", type="primary"):
                with st.spinner("Reprocessing..."):
                    pipeline.reprocess_batch(batch_id, method)
                st.session_state.pop(preview_key, None)
                st.rerun()

        if preview_key in st.session_state:
            st.image(
                st.session_state[preview_key][:, :, ::-1],
                caption="Green boxes = what this method would detect",
            )

    st.divider()
