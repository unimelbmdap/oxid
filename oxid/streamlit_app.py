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

            fig = go.Figure()

            # robust column detection
            x = None
            y = None

            for col in data.columns:
                c = col.lower()
                if x is None and ("field" in c or "x" == c):
                    x = data[col]
                if y is None and ("moment" in c or "mag" in c or "y" == c):
                    y = data[col]

            # fallback if structure is array-like
            if x is None or y is None:
                arr = data.values
                x = arr[:, 0]
                y = arr[:, 1]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=kind,
                )
            )

            fig.update_layout(
                title=f"{kind}: {path.name}",
                xaxis_title="Field",
                yaxis_title="Moment",
            )

            figs.append((kind, fig))

        except Exception as e:
            st.warning(f"Failed plotting {kind}: {e}")

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

gradients = st.sidebar.checkbox("Use Gradients", False)

mode = st.sidebar.selectbox("Plot Mode", ["lines+markers", "lines", "markers"])


# =========================
# UPLOAD
# =========================
st.header("Upload Data")

uploaded_files = st.file_uploader(
    "Upload .dat files",
    accept_multiple_files=True,
)

groups = {}

if use_hysteresis:
    groups["hysteresis"] = None
if use_rtsirm:
    groups["rtsirm"] = None
if use_zfcfc:
    groups["zfcfc"] = None


if uploaded_files:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:

        for f in uploaded_files:
            path = Path(tmpdir) / f.name
            path.write_bytes(f.getbuffer())

            kind = classify_file(path)

            if kind in groups:
                groups[kind] = path

    st.session_state.file_groups = groups

    st.subheader("Detected files")

    for k, v in groups.items():
        if v:
            st.success(f"{k}: {v.name}")
        else:
            st.warning(f"{k}: missing")


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