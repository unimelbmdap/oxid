from pathlib import Path
import subprocess
import tempfile
import pandas as pd


def run_oxid_embed(
    df: pd.DataFrame,
    components: int = 2):
    print("=== WRAPPER STARTED")
    print(df.head())

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

        print("COMMAND:")
        print(" ".join(cmd))
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        print("RETURN CODE:", result.returncode)
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        
        if result.returncode != 0:
            raise RuntimeError(
                f"OxID CLI failed:\n\n{result.stderr}"
            )

        return pd.read_csv(output_csv)