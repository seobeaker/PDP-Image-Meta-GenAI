"""
app.py — Typo Brand Image SEO Generator (Streamlit port of the Colab notebook)

Run locally:
    streamlit run app.py

Three modes, one per tab:
  1. Process now   - quick synchronous test on a few rows.
  2. Submit batch   - sends the whole sheet to the OpenAI Batch API (50% cheaper).
  3. Collect batch  - paste a Batch ID later to download results.
"""

import json
import time

import streamlit as st
from openai import OpenAI

import core

st.set_page_config(page_title="Image -> SEO generator", page_icon=":material/photo_camera:", layout="wide")

# ------------- SESSION STATE -------------
st.session_state.setdefault("last_batch_id", "")
st.session_state.setdefault("process_now_df", None)
st.session_state.setdefault("collect_df", None)
st.session_state.setdefault("collect_retry", [])

# ------------- SIDEBAR -------------
with st.sidebar:
    st.header("Settings")

    default_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input(
        "OpenAI API key", value=default_key, type="password", placeholder="sk-...",
        help="Stored only for this session. Prefer setting OPENAI_API_KEY in .streamlit/secrets.toml.",
    )
    model = st.selectbox("Model", core.MODEL_CHOICES, index=core.MODEL_CHOICES.index(core.DEFAULT_MODEL))
    st.caption("If a call fails with 'model not found', paste the exact id from your OpenAI dashboard.")
    brand = st.selectbox("Brand tone", list(core.BRAND_TONES.keys()),
                          index=list(core.BRAND_TONES.keys()).index(core.DEFAULT_BRAND))

    st.divider()
    st.markdown(
        "**Your spreadsheet**\n"
        "- `url` (required) - image URL, or a product page (og:image is used as a fallback)\n"
        "- product id (optional) - a column named `ID`, `product id`, `product_id`, or `sku`\n"
        "- `brand` (optional) - blank uses the brand tone above\n"
        "- `tone` (optional) - free-text style note, e.g. *luxe and minimal*"
    )

st.title("Image -> SEO generator")
st.caption("Turn a spreadsheet of product-image URLs into brand-tuned SEO copy: title, meta description, H1, short copy, alt text.")

tab_now, tab_submit, tab_collect = st.tabs(["Process now", "Submit batch", "Collect batch"])


def _require_key() -> bool:
    if not api_key.strip():
        st.warning("Enter your OpenAI API key in the sidebar first.")
        return False
    return True


def _download_buttons(df, key_prefix: str):
    st.dataframe(df.head(10), use_container_width=True)
    c1, c2 = st.columns(2)
    ts = time.strftime("%Y%m%d-%H%M%S")
    c1.download_button(
        "Download CSV", data=core.frame_to_csv_bytes(df),
        file_name=f"image_seo_{ts}.csv", mime="text/csv", key=f"{key_prefix}_csv",
    )
    c2.download_button(
        "Download XLSX", data=core.frame_to_xlsx_bytes(df),
        file_name=f"image_seo_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"{key_prefix}_xlsx",
    )


# ------------- MODE 1: PROCESS NOW -------------
with tab_now:
    st.subheader("Process now")
    st.caption("Great for testing a few rows before a big batch. Full price, instant.")

    pn_file = st.file_uploader("Upload sheet (.xlsx / .xls / .csv)", type=["xlsx", "xls", "csv"], key="pn_file")
    pn_limit = st.number_input("Max rows", min_value=1, value=5, step=1, key="pn_limit")

    if st.button("Process now", type="primary", key="pn_btn"):
        if _require_key() and pn_file is not None:
            try:
                df = core.read_table(pn_file)
            except Exception as e:
                st.error(f"Could not read the sheet: {e}")
                df = None

            if df is not None:
                df = df.head(max(1, int(pn_limit)))
                client = OpenAI(api_key=api_key.strip())
                rows, warnings = [], []
                progress = st.progress(0.0, text="Processing rows...")
                for i, (_, r) in enumerate(df.iterrows(), start=1):
                    url = str(r["url"]).strip()
                    if url:
                        try:
                            ro = core.process_row_sync(
                                client, model, str(r["pid"]).strip(), url,
                                str(r["brand"]).strip(), str(r["tone"]).strip(), brand,
                            )
                            if ro:
                                rows.append(ro)
                            else:
                                warnings.append(f"No image resolved for: {url}")
                        except Exception as e:
                            warnings.append(f"Row failed ({url}): {e}")
                    progress.progress(i / len(df), text=f"Processing rows... ({i}/{len(df)})")
                    time.sleep(core.PAUSE)
                progress.empty()

                for w in warnings:
                    st.warning(w)

                if not rows:
                    st.error("No outputs. Check your URLs (direct image URLs work best).")
                else:
                    st.session_state["process_now_df"] = core.rows_to_frame(rows)
                    st.success(f"Done. Processed {len(rows)} row(s).")
        elif pn_file is None:
            st.warning("Please upload a sheet with a 'url' column.")

    if st.session_state["process_now_df"] is not None:
        _download_buttons(st.session_state["process_now_df"], "pn")


# ------------- MODE 2: SUBMIT BATCH -------------
with tab_submit:
    st.subheader("Submit batch")
    st.caption("Uploads the whole sheet to the OpenAI Batch API (50% cheaper, runs in the background up to 24h). "
               "Prints a Batch ID - copy it, you'll need it to collect results.")

    sb_file = st.file_uploader("Upload sheet (.xlsx / .xls / .csv)", type=["xlsx", "xls", "csv"], key="sb_file")

    if st.button("Submit batch", type="primary", key="sb_btn"):
        if _require_key() and sb_file is not None:
            try:
                df = core.read_table(sb_file)
            except Exception as e:
                st.error(f"Could not read the sheet: {e}")
                df = None

            if df is not None:
                client = OpenAI(api_key=api_key.strip())
                lines, skipped = [], []
                progress = st.progress(0.0, text=f"Resolving images for {len(df)} rows...")
                for i, (idx, r) in enumerate(df.iterrows(), start=1):
                    url = str(r["url"]).strip()
                    if url:
                        pid = str(r["pid"]).strip()
                        profile, brand_used, tone_override = core.resolve_tone_profile(
                            str(r["brand"]).strip(), str(r["tone"]).strip(), brand,
                        )
                        image_ref = core.resolve_image_for_vision(url)
                        if not image_ref:
                            skipped.append((pid, url))
                        else:
                            lines.append(core.build_batch_line(
                                idx, pid, url, image_ref, brand_used, profile, tone_override, model,
                            ))
                    progress.progress(i / len(df), text=f"Resolving images... ({i}/{len(df)})")
                progress.empty()

                if not lines:
                    st.error("No images could be resolved - nothing to submit.")
                else:
                    jsonl_bytes = ("\n".join(json.dumps(ln, ensure_ascii=False) for ln in lines) + "\n").encode("utf-8")
                    up = client.files.create(file=("batch_input.jsonl", jsonl_bytes), purpose="batch")
                    batch = client.batches.create(
                        input_file_id=up.id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                        metadata={"app": "image-seo-v5"},
                    )
                    st.session_state["last_batch_id"] = batch.id
                    st.success("Batch submitted.")
                    st.code(batch.id, language=None)
                    st.write(f"Rows queued: **{len(lines)}** | Skipped (no image): **{len(skipped)}**")
                    st.info("Copy the Batch ID above. Paste it into the 'Collect batch' tab later - "
                            "it works even in a brand-new session.")
                    if skipped:
                        with st.expander(f"Skipped rows ({len(skipped)}) - no image found"):
                            for pid, u in skipped[:50]:
                                st.write(f"- [{pid}] {u}")
                            if len(skipped) > 50:
                                st.write(f"... and {len(skipped) - 50} more")
        elif sb_file is None:
            st.warning("Please upload a sheet with a 'url' column.")


# ------------- MODE 3: COLLECT BATCH -------------
with tab_collect:
    st.subheader("Collect batch")
    st.caption("Paste your Batch ID and API key, then check status / download results.")

    cb_id = st.text_input("Batch ID", value=st.session_state["last_batch_id"], placeholder="batch_...", key="cb_id")

    if st.button("Check / collect", type="primary", key="cb_btn"):
        if _require_key():
            bid = cb_id.strip()
            if not bid:
                st.warning("Please paste your Batch ID.")
            else:
                client = OpenAI(api_key=api_key.strip())
                try:
                    b = client.batches.retrieve(bid)
                except Exception as e:
                    st.error(f"Could not find that batch: {e}")
                    b = None

                if b is not None:
                    st.write(f"Status: **{b.status}**")
                    rc = getattr(b, "request_counts", None)
                    if rc:
                        st.write(f"Progress: {rc.completed}/{rc.total} done, {rc.failed} failed")

                    if b.status in ("validating", "in_progress", "finalizing"):
                        st.info("Not ready yet - come back later and click again.")
                    elif b.status in ("failed", "cancelled", "expired") and not b.output_file_id:
                        st.error(f"Batch ended without output ({b.status}).")
                    else:
                        meta_map = {}
                        if b.input_file_id:
                            for raw in client.files.content(b.input_file_id).text.splitlines():
                                if not raw.strip():
                                    continue
                                cid, meta, img = core.parse_input_line_meta(json.loads(raw))
                                meta_map[cid] = {"meta": meta, "image": img}

                        rows, retry = [], []
                        out_text = client.files.content(b.output_file_id).text
                        for raw in out_text.splitlines():
                            if not raw.strip():
                                continue
                            cid, data, err = core.parse_output_line(json.loads(raw))
                            info = meta_map.get(cid, {})
                            meta = info.get("meta", {})
                            pid = meta.get("id", "")
                            url = meta.get("url", cid)
                            row_brand = meta.get("brand", "")
                            profile = core.BRAND_TONES.get(row_brand, core.BRAND_TONES[core.DEFAULT_BRAND])
                            if err or not str(data.get("title", "")).strip():
                                retry.append((pid, url))
                                continue
                            clean = core.postprocess(data, profile)
                            rows.append(core.RowOut(
                                pid=pid, url=url, used_image=info.get("image", ""),
                                title=clean["title"], meta_description=clean["meta_description"],
                                h1=clean["h1"], short_copy=clean["short_copy"], alt_text=clean["alt_text"],
                                attrs_json=json.dumps(data.get("attrs", {}), ensure_ascii=False),
                                warnings=clean["warnings"],
                            ))

                        st.session_state["collect_retry"] = retry
                        if not rows:
                            st.warning("No usable rows returned.")
                            st.session_state["collect_df"] = None
                        else:
                            st.session_state["collect_df"] = core.rows_to_frame(rows)
                            st.success(f"Done. {len(rows)} row(s) collected.")

    if st.session_state["collect_df"] is not None:
        _download_buttons(st.session_state["collect_df"], "cb")

    retry = st.session_state.get("collect_retry") or []
    if retry:
        with st.expander(f"{len(retry)} row(s) came back empty (image likely not fetchable) - re-run via 'Process now'"):
            for pid, u in retry[:50]:
                st.write(f"- [{pid}] {u}")
            if len(retry) > 50:
                st.write(f"... and {len(retry) - 50} more")
