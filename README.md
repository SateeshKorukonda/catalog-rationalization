
# Catalogue Rationalization – Streamlit Demo (Optimised)

This fork replaces the single “Run full pipeline” button with stage‑by‑stage controls and Streamlit caches,
making the app **one to two orders of magnitude faster** on second runs.

## Quick start (local)

```bash
git clone <your‑repo>
cd <your‑repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What changed?

| Area | Old behaviour | New behaviour |
|------|---------------|---------------|
| **Execution** | `runpy.run_path()` runs the whole notebook every click | Each stage is a cached function; after the first run heavy work is skipped automatically |
| **UI** | Single button | Five clearly labelled sections & buttons (+ sidebar for data size / upload) |
| **Logs** | One giant code block | Logs per stage inside collapsible expanders |
| **Performance** | Minutes on every click | Seconds after initial cache |
| **Robustness** | No checks | Upload support, missing‑step warnings |

## Caching strategy

- **`@st.cache_data`** for deterministic DataFrames (synthetic generation, preprocessing).
- **`@st.cache_resource`** for training pipelines (Random Forest, K‑Means, XGBoost).  
  *The cache persists for ~24 h on Streamlit Community Cloud.*

## Next steps

- Replace synthetic data with your production dataset.
- Swap `st.code` logs for richer visualisations (charts, metrics).
- Use `st.file_uploader` for model artefacts and a `<download>` link for outputs.

---
© 2025 by ChatGPT.
