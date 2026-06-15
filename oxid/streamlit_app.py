import streamlit as st
from pathlib import Path
from collections import defaultdict
import pandas as pd

from read_magic import read_magic
    
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

if "file_groups" not in st.session_state:
    st.session_state.file_groups = {
        "hysteresis": [],
        "rtsirm": [],
        "zfcfc": [],
    }

if "embedding" not in st.session_state:
    st.session_state.embedding = None


# =========================
# FILE CLASSIFICATION
# =========================

def classify_file(path: Path):
    stem = path.stem.lower()

    if any(x in stem for x in ["rtsirm", "rt-sirm"]):
        return "rtsirm"

    if any(x in stem for x in ["zfcfc", "zfc-fc"]):
        return "zfcfc"

    if any(x in stem for x in ["hys", "hysteresis", "hyst"]):
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
# EMBEDDING DATAFRAME
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
# PIPELINE
# =========================

def run_pipeline(groups, upload_dir, use_hysteresis, use_rtsirm, use_zfcfc):

    df = build_embedding_dataframe(upload_dir, groups)

    if len(df) < 2:
        raise ValueError("UMAP requires at least two samples.")

    # =========================
    # STEP 2 — FILTER HERE (IMPORTANT)
    # =========================
    if not use_hysteresis:
        df["Hysteresis"] = None

    if not use_rtsirm:
        df["RTSIRM"] = None

    if not use_zfcfc:
        df["ZFCFC"] = None

    # OR better (cleaner hard filter):
    # df = only keep columns corresponding to selected toggles

    vectors = build_feature_vectors(
        df,
        hysteresis=use_hysteresis,
        rtsirm=use_rtsirm,
        zfcfc=use_zfcfc,
        points=250,
        features=20,
        include_normalized=True,
        include_unnormalized=True,
    )

    n_neighbors = min(15, max(2, len(vectors) - 1))

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
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="OxID",
    page_icon="🧲",
    layout="wide",
)

st.title("🧲 OxID Dashboard")


# =========================
# SIDEBAR
# =========================

st.sidebar.header("Controls")

use_hysteresis = st.sidebar.checkbox("Hysteresis", True)
use_rtsirm = st.sidebar.checkbox("RT-SIRM", True)
use_zfcfc = st.sidebar.checkbox("ZFC-FC", True)


# =========================
# UPLOAD
# =========================

st.header("Upload Data")

uploaded_files = st.file_uploader(
    "Upload .dat or MagIC files",
    accept_multiple_files=True,
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ------------------------
# SESSION STATE INIT (ONCE ONLY)
# ------------------------
if "magic_processed" not in st.session_state:
    st.session_state.magic_processed = set()

if "file_groups" not in st.session_state:
    st.session_state.file_groups = {
        "hysteresis": [],
        "rtsirm": [],
        "zfcfc": [],
    }

# ------------------------
# RESET GROUPS EACH RUN (IMPORTANT FOR STREAMLIT)
# ------------------------
st.session_state.file_groups = {
    "hysteresis": [],
    "rtsirm": [],
    "zfcfc": [],
}

# ------------------------
# PROCESS UPLOADS
# ------------------------
if uploaded_files:

    for uploaded_file in uploaded_files:

        path = UPLOAD_DIR / uploaded_file.name

        # Always overwrite local copy (keeps UI consistent)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ---------------- MAGIC FILES ----------------
        if path.suffix.lower() in [".txt", ".mag", ".magic"]:

            if path.name not in st.session_state.magic_processed:

                st.info(f"Processing MagIC file: {path.name}")

                outputs = read_magic(
                    path=str(path),
                    output_dir=UPLOAD_DIR,
                )

                st.session_state.magic_processed.add(path.name)

                if outputs:
                    for out in outputs:
                        kind = classify_file(out)

                        if kind in st.session_state.file_groups:
                            st.session_state.file_groups[kind].append(out)

            continue

        # ---------------- NORMAL FILES ----------------
        kind = classify_file(path)

        if kind in st.session_state.file_groups:
            st.session_state.file_groups[kind].append(path)

# =========================
# BUTTONS
# =========================

col1, col2 = st.columns(2)

run_clicked = col1.button("🚀 Run OxID")
plot_raw_clicked = col2.button("📈 Plot Raw Data")


# =========================
# RAW PLOTS
# =========================

if plot_raw_clicked:

    groups = st.session_state.file_groups

    if not groups or all(len(v) == 0 for v in groups.values()):
        st.error("Upload files first")
        st.stop()

    figs = plot_raw_files(groups)

    if not figs:
        st.warning("No valid plots generated.")
        st.stop()

    st.header("Raw Data")

    for name, fig in figs:
        st.subheader(name)
        st.plotly_chart(fig, use_container_width=True)


# =========================
# RUN PIPELINE
# =========================

if run_clicked:

    groups = st.session_state.file_groups

    if not groups:
        st.error("Upload files first.")
        st.stop()

    try:
        with st.spinner("Running UMAP..."):
            embedding, df = run_pipeline(
    groups,
    UPLOAD_DIR,
    use_hysteresis=use_hysteresis,
    use_rtsirm=use_rtsirm,
    use_zfcfc=use_zfcfc,
)

        st.session_state.embedding = {
            "coords": embedding,
            "df": df,
        }

        st.success(f"OxID complete ({len(df)} samples)")

    except Exception as e:
        st.exception(e)


# =========================
# UMAP RESULTS
# =========================

st.divider()
st.header("UMAP Embedding")

if st.session_state.embedding is None:
    st.info("Upload files and click 'Run OxID'.")
else:

    fig = plot_components(
        st.session_state.embedding["coords"],
        st.session_state.embedding["df"],
        title="OxID UMAP Projection",
        color_column="Group",
        show=False,
    )

    st.plotly_chart(fig, use_container_width=True)

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
        )

        st.session_state.embedding["df"] = editable_df


# =========================
# ABOUT
# =========================

with st.expander("About"):
    st.write(
        """
OxID is a scientific tool for analysing magnetic mineral datasets using hysteresis,
RT-SIRM and ZFC-FC measurements.
"""
    )