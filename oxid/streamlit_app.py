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


st.sidebar.subheader("Iron Oxides")

magnetite = st.sidebar.checkbox("Magnetite", value=True)
hematite = st.sidebar.checkbox("Hematite", value=True)
goethite = st.sidebar.checkbox("Goethite", value=True)
maghemite = st.sidebar.checkbox("Maghemite", value=True)
algoethite = st.sidebar.checkbox("Al-Goethite", value=True)


# ---------------------------
# FILE UPLOADS
# ---------------------------
st.header("Data Upload")

hysteresis_file = st.file_uploader("Hysteresis file")
rtsirm_file = st.file_uploader("RT-SIRM file")
zfcfc_file = st.file_uploader("ZFC-FC file")
MagIC_file = st.file_uploader("MagIC file (optional)")



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