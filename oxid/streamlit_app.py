import streamlit as st
from pathlib import Path

import plotly.graph_objects as go

from data import (
    data_files_list,
    iron_oxides_list,
    collate_results,
    Hysteresis,
    RTSIRM,
    ZFCFC,
)

from features import (
    build_feature_vectors,
    dimensionality_reduction,
)

from viz import plot_inputs as plot_inputs_viz
from viz import plot_components


# =========================
# SESSION STATE
# =========================
if "results" not in st.session_state:
    st.session_state.results = None

if "embedding" not in st.session_state:
    st.session_state.embedding = None

if "file_groups" not in st.session_state:
    st.session_state.file_groups = None


# =========================
# CLASSIFIER
# =========================
def classify_file(path: Path):
    name = path.name.lower()

    if "hys" in name:
        return "hysteresis"
    if "rtsirm" in name or "rt-sirm" in name:
        return "rtsirm"
    if "zfc" in name or "fc" in name:
        return "zfcfc"

    return "unknown"


# =========================
# RAW PLOT (Plotly + CLI-consistent)
# =========================
from data import Hysteresis, RTSIRM, ZFCFC
from viz import plot_moment


def plot_raw_files(groups):
    figs = []

    for kind, path in groups.items():
        if path is None:
            continue

        try:
            if kind == "hysteresis":
                data = Hysteresis(path)

            elif kind == "rtsirm":
                data = RTSIRM(path)

            elif kind == "zfcfc":
                data = ZFCFC(path)

            else:
                continue

            fig = plot_moment(
                data,
                title=path.name,
                show=False,
                output=None,
            )

            figs.append((kind, fig))

        except Exception as e:
            st.error(f"Failed plotting {kind}: {e}")

    return figs


# =========================
# PIPELINE WRAPPER (Streamlit-side)
# =========================
def run_pipeline(groups, gradients):
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

    import pandas as pd
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
        n_components=2,
        force=True,
    )

    return observed, basis_functions, regimes, iron_oxides, embedding, df


# =========================
# UI
# =========================
st.set_page_config(page_title="OxID", page_icon="🧲", layout="wide")

st.title("🧲 OxID Dashboard")
st.markdown("Upload magnetic datasets and analyse hysteresis, RT-SIRM, and ZFC-FC behaviour.")


# =========================
# SIDEBAR
# =========================
st.sidebar.header("Controls")

use_hysteresis = st.sidebar.checkbox("Hysteresis", True)
use_rtsirm = st.sidebar.checkbox("RT-SIRM", True)
use_zfcfc = st.sidebar.checkbox("ZFC-FC", True)

mode = st.sidebar.selectbox("Plot Mode", ["lines+markers", "lines", "markers"])


# =========================
# UPLOAD
# =========================
st.header("Upload Data")

uploaded_files = st.file_uploader(
    "Upload .dat files",
    accept_multiple_files=True,
)

# Persistent upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

groups = {}

if use_hysteresis:
    groups["hysteresis"] = []

if use_rtsirm:
    groups["rtsirm"] = []

if use_zfcfc:
    groups["zfcfc"] = []

if uploaded_files:

    for uploaded_file in uploaded_files:

        path = UPLOAD_DIR / uploaded_file.name

        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_type = classify_file(path)

        if file_type in groups:
            groups[file_type].append(path)

    st.session_state.file_groups = groups

    st.subheader("Detected Files")

    for file_type, files in groups.items():

        if len(files) > 0:

            st.success(
                f"{file_type}: {len(files)} file(s) detected"
            )

            for file in files:
                st.write(f"• {file.name}")

        else:
            st.warning(f"{file_type}: missing")
            
# =========================
# BUTTONS
# =========================
col1, col2 = st.columns(2)

run_clicked = col1.button("🚀 Run OxID")
plot_raw_clicked = col2.button("📈 Plot raw data")


# =========================
# RAW PLOT MODE
# =========================
if plot_raw_clicked:

    groups = st.session_state.file_groups

    if not groups:
        st.error("Upload files first")
        st.stop()

    figs = plot_raw_files(groups)

    if not figs:
        st.warning("No valid plots generated")
    else:
        for kind, fig in figs:
            st.subheader(kind)
            st.plotly_chart(fig, use_container_width=True)


# =========================
# RUN PIPELINE
# =========================
if run_clicked:

    groups = st.session_state.file_groups

    if not groups:
        st.error("Upload files first")
        st.stop()

    observed, basis, regimes, iron_oxides, embedding, df = run_pipeline(
        groups,
        gradients=gradients,
    )

    st.session_state.results = {
        "observed": observed,
        "basis": basis,
        "regimes": regimes,
        "iron_oxides": iron_oxides,
    }

    st.session_state.embedding = {
        "coords": embedding,
        "df": df,
    }

    st.success("OxID complete")


# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["Overview", "UMAP"])


with tab1:
    st.header("Overview")

    if st.session_state.results is None:
        st.info("Run OxID to generate results.")
    else:
        fig = plot_inputs_viz(
            st.session_state.results["observed"],
            st.session_state.results["basis"],
            st.session_state.results["regimes"],
            st.session_state.results["iron_oxides"],
            rescale=True,
            show=False,
            output=None,
            mode=mode,
        )

        st.plotly_chart(fig, use_container_width=True)


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


# =========================
# FOOTER
# =========================
with st.expander("About"):
    st.write(
        "OxID is a scientific tool for analysing magnetic mineral datasets."
    )