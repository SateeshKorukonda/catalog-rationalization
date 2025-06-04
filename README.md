
# Catalogue Rationalization – Streamlit Demo

This repository contains a Streamlit app that wraps the original Jupyter notebook
`catalogue_rationalization_code_LoyLogic.ipynb`.

## File structure

```
.
├── catalogue_rationalization_code_LoyLogic.ipynb   # original notebook
├── catalogue_rationalization_code_LoyLogic.py      # auto‑converted notebook (do not edit)
├── streamlit_app.py                                # Streamlit entry point
├── requirements.txt                                # Python dependencies
└── README.md
```

## Quickstart (local)

```bash
# 1. Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the Streamlit app
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Push all files to a **public GitHub repository**.
2. Go to <https://share.streamlit.io>, click **“New app”** and select your repo.
3. For *“Main file path”* enter `streamlit_app.py`.
4. Click **Deploy**. 🎉

The notebook will execute when you press **Run full pipeline** inside the app.

## Tips for a smoother app

* Refactor heavy computations into functions decorated with `@st.cache_data`
  or `@st.cache_resource` so they don’t re‑run on every user interaction.
* Replace `print` with Streamlit calls (`st.write`, `st.dataframe`, `st.pyplot`, …)
  to stream rich, interactive output directly to the browser.
* If you have any **API keys or secrets**, add them via **Settings ➜ Secrets**
  in Streamlit Cloud and access with `st.secrets["MY_KEY"]`.
