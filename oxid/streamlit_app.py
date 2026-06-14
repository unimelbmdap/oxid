import streamlit as st
from pathlib import Path
import tempfile

from data import data_files_list, iron_oxides_list, collate_results
from viz import plot_inputs, plot_components
from features import build_feature_vectors, dimensionality_reduction
import pandas as pd


# ---------------------------
# SESSION STATE INIT
# ---------------------------
if "results" not in st.session_state:
    st.session_state.results = None

if "embedding" not in st.session_state:
    st.session_state.embedding = None

if "file_groups" not in st.session_state:
    st.session_state.file_groups = None


# ---------------------------
# CLASSIFIER
# ---------------------------
def classify_file(path):
    name = path.name.lower()

    if "hys" in name or "hysteresis" in name:
        return "hysteresis"
    if "rtsirm" in name or "rt-sirm" in name:
        return "rtsirm"
    if "zfc" in name or "fc" in name:
        return "zfcfc"

    return "unknown"


# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Oxid",
    page_icon="🧲",
    layout="wide",
)

st.title("🧲 OxID Dashboard")
st.markdown("Upload magnetic datasets and visualise hysteresis, RT-SIRM, and ZFC-FC behaviour.")


# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("Controls")

use_hysteresis = st.sidebar.checkbox("Hysteresis", value=True)
use_rtsirm = st.sidebar.checkbox("RT-SIRM", value=True)
use_zfcfc = st.sidebar.checkbox("ZFC-FC", value=True)

if not (use_hysteresis or use_rtsirm or use_zfcfc):
    st.sidebar.error("Select at least one data type.")
    st.stop()

rescale = st.sidebar.checkbox("Rescale", value=True)
gradients = st.sidebar.checkbox("Use Gradients", value=False)

mode = st.sidebar.selectbox(
    "Plot Mode",
    ["lines+markers", "lines", "markers"],
)


# ---------------------------
# FILE UPLOAD + CLASSIFICATION
# ---------------------------
st.header("Upload Data")

uploaded_files = st.file_uploader(
    "Upload .dat files",
    accept_multiple_files=True
)

groups = {}

if use_hysteresis:
    groups["hysteresis"] = None
if use_rtsirm:
    groups["rtsirm"] = None
if use_zfcfc:
    groups["zfcfc"] = None


if uploaded_files:

    with tempfile.TemporaryDirectory() as tmpdir:

        for f in uploaded_files:
            path = Path(tmpdir) / f.name
            path.write_bytes(f.getbuffer())

            file_type = classify_file(path)

            if file_type in groups:
                groups[file_type] = path

    st.session_state.file_groups = groups

    st.subheader("Detected files")

    for k, v in groups.items():
        if v:
            st.success(f"{k}: {v.name}")
        else:
            st.warning(f"{k}: missing")


# ---------------------------
# RUN PIPELINE BUTTON
# ---------------------------
col1, col2 = st.columns(2)

run_clicked = col1.button("🚀 Run OxID")
plot_raw_clicked = col2.button("📈 Plot raw data")

if plot_raw_clicked:

    groups = st.session_state.file_groups

    if not groups:
        st.error("Upload files first")
        st.stop()

    fig = plot_raw_files(groups)

    st.pyplot(fig)
    
    # -------------------------
    # DATA PIPELINE
    # -------------------------
    data_files = data_files_list(
        groups.get("hysteresis"),
        groups.get("rtsirm"),
        groups.get("zfcfc"),
    )

    iron_oxides = iron_oxides_list(True, True, True, True, True)

    observed, basis_functions, regimes, datatypes = collate_results(
        data_files,
        iron_oxides,
        gradients=gradients,
    )

    st.session_state.results = {
        "observed": observed,
        "basis": basis_functions,
        "regimes": regimes,
        "iron_oxides": iron_oxides,
    }

    # -------------------------
    # UMAP PIPELINE
    # -------------------------
    df = pd.DataFrame({"Name": regimes})

    vectors = build_feature_vectors(
        df,
        hysteresis=True,
        rtsirm=True,
        zfcfc=True,
        points=250,
        features=20,
        include_normalized=True,
        include_unnormalized=True,
    )

    embedding = dimensionality_reduction(
        vectors,
        n_neighbors=15,
        min_dist=0.1,
        seed=0,
        reducer_path=None,
        n_components=2,
        force=True,
    )

    st.session_state.embedding = {
        "coords": embedding,
        "df": df,
    }

    st.success("OxID complete: plots + UMAP generated")

# ---------------------------
# PLOTTING FUNCTIONS
# ---------------------------
import matplotlib.pyplot as plt

def plot_raw_files(groups):
    fig, ax = plt.subplots()

    for key, path in groups.items():
        if path is None:
            continue

        try:
            data = pd.read_csv(path, delim_whitespace=True, header=None)

            # assume first two columns are x/y
            x = data.iloc[:, 0]
            y = data.iloc[:, 1]

            ax.plot(x, y, label=key)

        except Exception as e:
            st.warning(f"Could not plot {key}: {e}")

    ax.set_title("Raw Data Plot")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    return fig

# ---------------------------
# TABS
# ---------------------------
tab1, tab2 = st.tabs(["Overview", "UMAP"])


# ---------------------------
# OVERVIEW TAB
# ---------------------------
with tab1:

    st.header("Overview")

    if st.session_state.results is None:
        st.info("Run OxID to generate results.")
    else:
        fig = plot_inputs(
            st.session_state.results["observed"],
            st.session_state.results["basis"],
            st.session_state.results["regimes"],
            st.session_state.results["iron_oxides"],
            rescale=True,
            show=False,
            output=None,
            mode="lines+markers",
        )

        st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# UMAP TAB
# ---------------------------
with tab2:

    st.header("UMAP Embedding")

    if st.session_state.embedding is None:
        st.info("Run OxID to generate embeddings.")
    else:
        fig = plot_components(
            st.session_state.embedding["coords"],
            st.session_state.embedding["df"],
            title="UMAP Projection",
        )

        st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# ABOUT
# ---------------------------
with st.expander("About this tool"):
    st.write(
        """
        OxID is a scientific tool for analysing magnetic mineral data.
        Upload datasets to visualise hysteresis loops, RT-SIRM, and ZFC-FC behaviour.
        """
    )