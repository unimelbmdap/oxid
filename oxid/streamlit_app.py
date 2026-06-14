import streamlit as st
from pathlib import Path
import tempfile

from data import (
    data_files_list,
    iron_oxides_list,
    collate_results,
    RTSIRM,
    ZFCFC,
    Hysteresis,
)

from viz import plot_inputs


# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Oxid",
    page_icon="🧲",
    layout="wide",
)

st.title("🧲 Oxid Scientific Dashboard")
st.markdown("Upload magnetic datasets and visualise hysteresis, RT-SIRM, and ZFC-FC behaviour.")


# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
st.sidebar.header("Controls")

rescale = st.sidebar.checkbox("Rescale", value=True)
gradients = st.sidebar.checkbox("Use Gradients", value=False)

mode = st.sidebar.selectbox(
    "Plot Mode",
    ["lines+markers", "lines", "markers"],
)



# ---------------------------
# FILE UPLOADS
# ---------------------------
st.header("Data Upload")

mode = st.radio(
    "Upload mode",
    ["Multiple files", "ZIP folder"],
)

file_paths = []

with tempfile.TemporaryDirectory() as tmpdir:

    if mode == "Multiple files":

        uploaded_files = st.file_uploader(
            "Upload .dat files",
            accept_multiple_files=True
        )

        if uploaded_files:
            for f in uploaded_files:
                path = Path(tmpdir) / f.name
                path.write_bytes(f.getbuffer())
                file_paths.append(path)

    else:

        zip_file = st.file_uploader("Upload ZIP file", type="zip")

        if zip_file:
            import zipfile
            with zipfile.ZipFile(zip_file) as z:
                z.extractall(tmpdir)

            file_paths = list(Path(tmpdir).rglob("*"))
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
# OPTIONAL INFO SECTION
# ---------------------------
with st.expander("About this tool"):
    st.write(
        """
        Oxid is a scientific tool for analysing magnetic mineral data.
        Upload datasets to visualise hysteresis loops, RT-SIRM, and ZFC-FC behaviour.
        """
    )