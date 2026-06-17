from pathlib import Path
import subprocess
import tempfile
import pandas as pd


def run_oxid_embed(
    df: pd.DataFrame,
    components: int = 2,
):
    """
    Run the existing OxID CLI embedding pipeline.
    """

    with tempfile.TemporaryDirectory() as tmp:

        tmp = Path(tmp)

        input_csv = tmp / "input.csv"
        output_csv = tmp / "embedding.csv"
        reducer_file = tmp / f"reducer-{components}.umap"

        df.to_csv(input_csv, index=False)

        cmd = [
            "oxid",
            "embed",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--reducer",
            str(reducer_file),
            "--components",
            str(components),
            "--force",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"OxID CLI failed:\n\n{result.stderr}"
            )

        return pd.read_csv(output_csv)