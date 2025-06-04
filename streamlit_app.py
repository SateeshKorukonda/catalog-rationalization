
import streamlit as st
from pathlib import Path
import runpy
import io
from contextlib import redirect_stdout

st.set_page_config(page_title="Catalogue Rationalization Demo",
                   page_icon="📊",
                   layout="wide")

st.title("📊 Catalogue Rationalization Demo")
st.write(
    """
    This Streamlit app wraps the original Jupyter notebook that demonstrates catalogue
    rationalization using data preprocessing, Random Forest, K‑Means clustering and XGBoost.

    **What can you do here?**
    1. Click **Run full pipeline** to execute every step of the notebook inside Streamlit.
    2. Once finished, review the printed logs and any result files the notebook creates
       (for example CSV or model artifacts).

    > *Tip:* If you refactor the code into functions like `load_data()`, `train_model()`,
    > and reuse `@st.cache_data`, the app will become much faster and interactive
    > (dropdowns, sliders, charts …).
    """
)

NOTEBOOK_PY = Path(__file__).parent / "catalogue_rationalization_code_LoyLogic.py"

if not NOTEBOOK_PY.exists():
    st.error("The converted notebook script is missing. Make sure it’s committed next to this file.")
    st.stop()

run_it = st.button("🚀 Run full pipeline", type="primary")

if run_it:
    st.toast("Execution started – grab a coffee ☕", icon="⏳")
    log_placeholder = st.empty()
    output = io.StringIO()
    with redirect_stdout(output):
        # Executes the script in a fresh global namespace
        runpy.run_path(str(NOTEBOOK_PY), run_name="__main__")
    log_placeholder.code(output.getvalue(), language="python")
    st.success("Notebook run complete!")
