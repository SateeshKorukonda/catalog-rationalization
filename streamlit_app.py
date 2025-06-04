
import streamlit as st
from pathlib import Path
import io
from contextlib import redirect_stdout
import pandas as pd

# Import functions from the converted notebook script
# They will run exactly as in the notebook, but we wrap them in Streamlit caches
from catalogue_rationalization_code_LoyLogic import (
    generate_catalog_data,
    preprocess_pipeline,
    rf_analysis_pipeline,
    kmeans_analysis_pipeline,
    tuned_xgb_pipeline,
)

st.set_page_config(
    page_title="Catalogue Rationalization Demo",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Catalogue Rationalization – Interactive Demo")

with st.expander("ℹ️ About this app", expanded=True):
    st.markdown(
        '''
        *This Streamlit interface lets you run the key stages of the original Jupyter notebook on‑demand, but **without re‑executing every expensive step on each rerun**.*  
        Long‑running parts are cached automatically, so after the first run they feel instant.

        **Workflow**

        1. **Generate** (or upload) a synthetic product catalogue.
        2. **Pre‑process** it (missing values, encoding, scaling).
        3. Run **Random Forest feature importance**.
        4. Run **K‑Means clustering**.
        5. Run the **tuned XGBoost classifier**.

        Each stage streams its console logs under a collapsible section, so you can inspect what happened without digging into Python files.
        '''
    )

# -------------------------------------------------------
# 1️⃣  DATA STAGE
# -------------------------------------------------------
st.header("1️⃣ Data source")

# Sidebar controls
st.sidebar.header("Data options")
n_products = st.sidebar.slider(
    "Number of synthetic products", min_value=500, max_value=10_000, value=3_000, step=500
)

uploaded = st.sidebar.file_uploader(
    "…or upload a CSV to use your own catalogue (optional)",
    type=["csv"],
    key="upload",
)

@st.cache_data(show_spinner=False)
def load_synthetic(n_rows: int) -> pd.DataFrame:
    """Generate and return a synthetic product catalogue."""
    return generate_catalog_data(n_rows)

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df_raw):,} rows from upload.")
else:
    if st.button("🚀 Generate synthetic data"):
        with st.spinner("Generating synthetic catalogue…"):
            df_raw = load_synthetic(n_products)
            st.success(f"Synthetic data ready ({len(df_raw):,} rows)")

if "df_raw" in locals():
    st.subheader("Raw sample")
    st.dataframe(df_raw.head())

# -------------------------------------------------------
# 2️⃣  PRE‑PROCESSING
# -------------------------------------------------------
st.header("2️⃣ Pre‑processing")

@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return preprocess_pipeline(df)

if st.button("⚙️ Run pre‑processing"):
    if "df_raw" not in locals():
        st.warning("Please generate or upload data first.")
    else:
        with st.spinner("Running preprocessing pipeline…"):
            df_pre = preprocess(df_raw)
            st.session_state["df_pre"] = df_pre

        st.success("Preprocessing finished and cached!")
        st.dataframe(df_pre.head())

# -------------------------------------------------------
# Helper to wrap heavy pipelines with caching + log capture
# -------------------------------------------------------
def cached_pipeline(fn, cache_name):
    """Return a Streamlit‑cached wrapper around *fn* that also captures stdout."""

    @st.cache_resource(show_spinner=False)
    def _runner(df):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = fn(df)
        log = stdout.getvalue()
        return result, log

    return _runner

rf_cached = cached_pipeline(rf_analysis_pipeline, "rf")
km_cached = cached_pipeline(kmeans_analysis_pipeline, "kmeans")
xgb_cached = cached_pipeline(tuned_xgb_pipeline, "xgb")

# -------------------------------------------------------
# 3️⃣  RANDOM FOREST FEATURE IMPORTANCE
# -------------------------------------------------------
st.header("3️⃣ Random Forest feature importance")

if st.button("🌲 Run Random Forest analysis"):
    if "df_pre" not in st.session_state:
        st.warning("Please run preprocessing first.")
    else:
        with st.spinner("Training Random Forest… this may take a minute on first run"):
            (rf_out, rf_log) = rf_cached(st.session_state["df_pre"])

        st.success("Random Forest analysis complete!")
        with st.expander("🔍 Logs – Random Forest"):
            st.code(rf_log, language="python")

# -------------------------------------------------------
# 4️⃣  K‑MEANS CLUSTERING
# -------------------------------------------------------
st.header("4️⃣ K‑Means clustering")

if st.button("📊 Run K‑Means clustering"):
    if "df_pre" not in st.session_state:
        st.warning("Please run preprocessing first.")
    else:
        with st.spinner("Running K‑Means…"):
            (km_out, km_log) = km_cached(st.session_state["df_pre"])

        st.success("K‑Means clustering complete!")
        with st.expander("🔍 Logs – K‑Means"):
            st.code(km_log, language="python")

# -------------------------------------------------------
# 5️⃣  Tuned XGBoost classifier
# -------------------------------------------------------
st.header("5️⃣ Tuned XGBoost classification")

if st.button("⚡ Run tuned XGBoost"):
    if "df_pre" not in st.session_state:
        st.warning("Please run preprocessing first.")
    else:
        with st.spinner("Training XGBoost… this may take several minutes on first run"):
            (xgb_out, xgb_log) = xgb_cached(st.session_state["df_pre"])

        st.success("XGBoost pipeline complete!")
        with st.expander("🔍 Logs – XGBoost"):
            st.code(xgb_log, language="python")

st.markdown("---")
st.caption("© 2025 · Streamlit demo refactored by ChatGPT · Caches persist ~24 h on Community Cloud.")
