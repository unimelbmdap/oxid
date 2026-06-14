import streamlit as st
from pathlib import Path
from collections import defaultdict
import pandas as pd

from data import Hysteresis, RTSIRM, ZFCFC

from features import (
    build_feature_vectors,
    dimensionality_reduction,
)

from viz import (
    plot_components,
    plot_moment,
)

# =========================
# SESSION STATE
# =========================

if "embedding" not in st.session_state:
    st.session_state.embedding = None

if "file_groups" not in st.session_state:
    st.session_state.file_groups = None


# =========================
# FILE CLASSIFICATION
# =========================

def classify_file(path: Path):

    stem = path.stem.lower()

    if stem.endswith("rtsirm" or "RTSIRM" or "RT-SIRM"):
        return "rtsirm"

    if stem.endswith("zfcfc" or "ZFCFC" or "ZFC-FC"):
        return "zfcfc"

    if stem.endswith("hyst" or "HYST" or "HYSTERESIS" or "HYS" or "hys"):
        return "hysteresis"

    return "unknown"


# =========================
# SAMPLE NAME EXTRACTION
# =========================

def sample_name_from_file(path: Path):

    stem = path.stem

    lower = stem.lower()

    if lower.endswith("_hys"):
        return stem[:-4]

    if lower.endswith("_rtsirm"):
        return stem[:-7]

    if lower.endswith("_zfcfc"):
        return stem[:-6]

    return stem


# =========================
# RAW PLOTS
# =========================

def plot_raw_files(groups):

    figs = []

    for kind, paths in groups.items():

        for path in paths:

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

                figs.append((path.name, fig))

            except Exception as e:
                st.error(f"Failed plotting {path.name}: {e}")

    return figs


# =========================
# BUILD OXID DATAFRAME
# =========================

def build_embedding_dataframe(upload_dir, groups):

    samples = defaultdict(dict)

    for path in groups.get("hysteresis", []):

        sample = sample_name_from_file(path)

        samples[sample]["Name"] = sample
        samples[sample]["Hysteresis"] = path.name

    for path in groups.get("rtsirm", []):

        sample = sample_name_from_file(path)

        samples[sample]["Name"] = sample
        samples[sample]["RTSIRM"] = path.name

    for path in groups.get("zfcfc", []):

        sample = sample_name_from_file(path)

        samples[sample]["Name"] = sample
        samples[sample]["ZFCFC"] = path.name

    rows = []

    for sample_name, row in samples.items():

        row["Group"] = "Unassigned"
        row["base_dir"] = str(upload_dir)

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# OXID PIPELINE
# =========================

def run_pipeline(groups, upload_dir):

    df = build_embedding_dataframe(
        upload_dir,
        groups,
    )

    if len(df) < 2:
        raise ValueError(
            "UMAP requires at least two samples."
        )

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

    if len(vectors) < 2:
        raise ValueError(
            "Unable to generate enough feature vectors for UMAP."
        )

    n_neighbors = min(
        15,
        max(2, len(vectors) - 1),
    )

    embedding = dimensionality_reduction(
        vectors,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        seed=0,
        n_components=2,
        force=True,
    )

    return embedding, df


# =========================
# PAGE
# =========================

st.set_page_config(
    page_title="OxID",
    page_icon="🧲",
    layout="wide",
)

st.title("🧲 OxID Dashboard")

st.markdown(
    """
Upload magnetic datasets and generate OxID UMAP embeddings.

Accepted naming conventions:

- Sample01_hys.dat
- Sample01_rtsirm.dat
- Sample01_zfcfc.dat

Samples may contain one, two, or all three measurement types.
"""
)

# =========================
# SIDEBAR
# =========================

st.sidebar.header("Controls")

use_hysteresis = st.sidebar.checkbox(
    "Hysteresis",
    True,
)

use_rtsirm = st.sidebar.checkbox(
    "RT-SIRM",
    True,
)

use_zfcfc = st.sidebar.checkbox(
    "ZFC-FC",
    True,
)

# =========================
# UPLOAD
# =========================

st.header("Upload Data")

uploaded_files = st.file_uploader(
    "Upload .dat files",
    accept_multiple_files=True,
)

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

        # =========================
        # MAGiC DETECTION
        # =========================
        if path.suffix.lower() in [".txt", ".mag", ".magic"]:

            st.info(f"Processing MagIC file: {path.name}")

            read_magic(
                path=str(path),
                output_dir=UPLOAD_DIR,
            )

            continue  # IMPORTANT: skip normal classification

        # =========================
        # NORMAL OXID FILES
        # =========================
        file_type = classify_file(path)

        if file_type in groups:
            groups[file_type].append(path)

# =========================
# BUTTONS
# =========================

col1, col2 = st.columns(2)

run_clicked = col1.button(
    "🚀 Run OxID",
    use_container_width=True,
)

plot_raw_clicked = col2.button(
    "📈 Plot Raw Data",
    use_container_width=True,
)

# =========================
# RAW PLOTS
# =========================

if plot_raw_clicked:

    groups = st.session_state.file_groups

    if not groups:

        st.error("Upload files first.")
        st.stop()

    figs = plot_raw_files(groups)

    if not figs:

        st.warning("No valid plots generated.")

    else:

        st.header("Raw Data")

        for name, fig in figs:

            st.subheader(name)

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

# =========================
# RUN OXID
# =========================

if run_clicked:

    groups = st.session_state.file_groups

    if not groups:

        st.error("Upload files first.")
        st.stop()

    try:

        with st.spinner(
            "Generating feature vectors and running UMAP..."
        ):

            embedding, df = run_pipeline(
                groups,
                UPLOAD_DIR,
            )

        st.session_state.embedding = {
            "coords": embedding,
            "df": df,
        }

        st.success(
            f"OxID complete ({len(df)} samples processed)"
        )

    except Exception as e:

        st.exception(e)

# =========================
# UMAP RESULTS
# =========================

st.divider()

st.header("UMAP Embedding")

if st.session_state.embedding is None:

    st.info(
        "Upload files and click 'Run OxID'."
    )

else:

    fig = plot_components(
    st.session_state.embedding = {
    "coords": embedding,
    "df": df,
}
    title="OxID UMAP Projection",
    color_column="Group",
    show=False,
)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

with st.expander("Sample Grouping", expanded=True):

    editable_df = st.data_editor(
        st.session_state.embedding["df"],
        column_config={
            "Group": st.column_config.TextColumn(
                "Group",
                help="Assign samples to groups",
            ),
        },
        disabled=[
            c for c in st.session_state.embedding["df"].columns
            if c != "Group"
        ],
        use_container_width=True,
        key="group_editor",
    )

    st.session_state.embedding["df"] = editable_df

# =========================
# ABOUT
# =========================

with st.expander("About"):

    st.write(
        """
OxID is a scientific tool for analysing
magnetic mineral datasets using hysteresis,
RT-SIRM and ZFC-FC measurements.
"""
    )