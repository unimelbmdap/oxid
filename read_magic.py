import pandas as pd
from pathlib import Path
import typer

app = typer.Typer()


@app.command()
def read_magic(
    path: str = typer.Argument(..., help="Path to the MAGIC data file"),
    output_dir: Path = typer.Argument(..., help="Directory to save extracted data files"),
):
    """
    Read and display contents of a MAGIC data file.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(path, 'r') as f:
        lines = f.readlines()
    
    speciments_start_index = next(i for i, line in enumerate(lines) if 'tab delimited	specimens' in line)
    speciments_end_index = next(i for i, line in enumerate(lines[speciments_start_index:]) if '>>>>>>>>>' in line)
    measurements_start_index = next(i for i, line in enumerate(lines) if 'tab delimited	measurements' in line)

    specimens_df = pd.read_csv(path, skiprows=speciments_start_index + 1, sep='\t', nrows=speciments_end_index - 2)

    
    measurements_df = pd.read_csv(path, skiprows=measurements_start_index + 1, sep='\t')

    MEASUREMENT_CODES = [
        'LP-ZFC',
        'LP-FC',
        # 'LP-HYS',
        'LP-CW-SIRM',
    ]
    measurements_df = measurements_df[measurements_df['method_codes'].isin(MEASUREMENT_CODES)].copy()

    measurements_df['Temperature (K)'] = measurements_df['treat_temp']
    measurements_df['Moment (emu)'] = measurements_df['magn_moment'] * 1000

    specimen_names = measurements_df['specimen'].unique()
    for specimen in specimen_names:
        print(f"Specimen: {specimen}")
        specimen_data = specimens_df[specimens_df.specimen == specimen]
        mass = specimen_data["weight"].values[0] * 1_000_000
        specimen_measurements_df = measurements_df[measurements_df['specimen'] == specimen]

        measurements = specimen_measurements_df['method_codes'].unique()
        for measurement in measurements:
            print(f"  Measurement: {measurement}")
            measurements_df = specimen_measurements_df[specimen_measurements_df['method_codes'] == measurement]
            
            new_path = output_dir/f"{specimen}_{measurement}.dat"
            new_path.write_text(f"INFO,{mass},SAMPLE_MASS\n[Data]\n")
            measurements_df[['Temperature (K)', 'Moment (emu)']].to_csv(new_path, index=False, mode="a", header=True)
            print(f"    Saved data to {new_path}")


if __name__ == "__main__":
    app()