import streamlit as st
from pathlib import Path
import tempfile

from data import data_files_list, iron_oxides_list, collate_results
from viz import plot_inputs

if "results" not in st.session_state:
    st.session_state.results = None

if "embedding" not in st.session_state:
    st.session_state.embedding = None

if "file_groups" not in st.session_state:
    st.session_state.file_groups = None

# ---------------------------
# Add Classifier
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
# SIDEBAR CONTROLS
# ---------------------------
st.sidebar.header("Controls")

st.sidebar.header("Data Types")

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
# FILE UPLOADS
# ---------------------------
st.header("Upload Data")

uploaded_files = st.file_uploader(
    "Upload .dat files",
    accept_multiple_files=True
)

if uploaded_files:

    with tempfile.TemporaryDirectory() as tmpdir:
for uploaded_file in uploaded_files:
            file_path = os.path.join(tmpdir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.success("Files uploaded successfully!")
# -----------------------
# SAVE + CLASSIFY FILES
# -----------------------

groups = {}

if use_hysteresis:
    groups["hysteresis"] = None

if use_rtsirm:
    groups["rtsirm"] = None

if use_zfcfc:
    groups["zfcfc"] = None

        for f in uploaded_files:
            path = Path(tmpdir) / f.name
            path.write_bytes(f.getbuffer())

            file_type = classify_file(path)

if file_type in groups:
    groups[file_type] = path

        # -----------------------
        # SHOW DETECTION RESULT
        # -----------------------
        st.subheader("Detected files")

        for k, v in groups.items():
            if v:
                st.success(f"{k}: {v.name}")
            else:
                st.warning(f"{k}: missing")

        # -----------------------
        # CONTINUE PIPELINE
        # -----------------------
        data_files = data_files_list(
            groups["hysteresis"],
            groups["rtsirm"],
            groups["zfcfc"],
        )

        iron_oxides = iron_oxides_list(
            True, True, True, True, True  # you can replace with UI later
        )

        observed, basis_functions, regimes, datatypes = collate_results(
            data_files,
            iron_oxides,
            gradients=gradients,
        )

        fig = plot_inputs(
            observed,
            basis_functions,
            regimes,
            iron_oxides,
            rescale=rescale,
            show=False,
            output=None,
            mode=mode,
        )

        st.plotly_chart(fig, use_container_width=True)
# ---------------------------
# MAIN ACTION
# ---------------------------
if st.button("Generate Plot"):

    with tempfile.TemporaryDirectory() as tmpdir:

        def save_upload(upload):
            if upload is None:
                return None
            path = Path(tmpdir) / upload.name
            path.write_bytes(upload.getbuffer())
            return path

        hys_path = save_upload(hysteresis_file)
        rtsirm_path = save_upload(rtsirm_file)
        zfcfc_path = save_upload(zfcfc_file)

        data_files = data_files_list(hys_path, rtsirm_path, zfcfc_path)

        iron_oxides = iron_oxides_list(
            goethite,
            hematite,
            magnetite,
            maghemite,
            algoethite,
        )

        observed, basis_functions, regimes, datatypes = collate_results(
            data_files,
            iron_oxides,
            gradients=gradients,
        )

        fig = plot_inputs(
            observed,
            basis_functions,
            regimes,
            iron_oxides,
            rescale=rescale,
            show=False,
            output=None,
            mode=mode,
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Run OxID
# ---------------------------
if st.button("🚀 Run OxID"):

    groups = st.session_state.file_groups

    if groups is None:
        st.error("Upload files first")
        st.stop()

    # -------------------------
    # STEP 1: DATA PIPELINE
    # -------------------------
    data_files = data_files_list(
        groups.get("hysteresis"),
        groups.get("rtsirm"),
        groups.get("zfcfc"),
    )

    iron_oxides = iron_oxides_list(
        True, True, True, True, True
    )

    observed, basis_functions, regimes, datatypes = collate_results(
        data_files,
        iron_oxides,
        gradients=False,
    )

    st.session_state.results = {
        "observed": observed,
        "basis": basis_functions,
        "regimes": regimes,
        "iron_oxides": iron_oxides,
    }

    # -------------------------
    # STEP 2: UMAP PIPELINE
    # -------------------------
    from features import build_feature_vectors, dimensionality_reduction
    import pandas as pd

    # rebuild dataframe for embedding
    # (you may already have this upstream in your CLI code)
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
# Overview Tab
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
# UMAP visualisation Tab
# ---------------------------
with tab2:

    st.header("UMAP Embedding")

    if st.session_state.embedding is None:
        st.info("Run OxID to generate embeddings.")
    else:
        from viz import plot_components

        fig = plot_components(
            st.session_state.embedding["coords"],
            st.session_state.embedding["df"],
            title="UMAP Projection",
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# OPTIONAL INFO SECTION
# ---------------------------
with st.expander("About this tool"):
    st.write(
        """
        Oxid is a scientific tool for analysing magnetic mineral data.
        Upload datasets to visualise hysteresis loops, RT-SIRM, and ZFC-FC behaviour.
        """
    )