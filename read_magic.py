import pandas as pd
from pathlib import Path
import typer

app = typer.Typer(pretty_exceptions_enable=False)

def write_measurement(output_dir: Path, specimen: str, measurement_type: str, mass: float, df: pd.DataFrame, x_column="Temperature (K)"):
    if df.empty:
        return

    # Check to see if 'Moment (A⋅m2/kg)' is non NaN
    if 'Moment (A⋅m2/kg)' in df.columns and not df['Moment (A⋅m2/kg)'].isnull().all():
        moment_column = 'Moment (A⋅m2/kg)'
    else:
        moment_column = 'Moment (emu)'

    new_path = output_dir/f"{specimen}_{measurement_type}.dat"
    new_path.write_text(f"INFO,{mass},SAMPLE_MASS\n[Data]\n")
    df[[x_column, moment_column]].to_csv(new_path, index=False, mode="a", header=True)
    print(f"    Saved data to {new_path}")


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
        'LP-HYS',
        'LP-CW-SIRM',
    ]

    available_codes = measurements_df['method_codes'].unique()
    zfc_code = None
    fc_code = None
    hys_code = None
    sirm_code = None
    compatible_codes = set()
    for code in available_codes:
        if 'LP-ZFC' in code:
            zfc_code = code
            compatible_codes.add(zfc_code)
        if 'LP-FC' in code:
            fc_code = code
            compatible_codes.add(fc_code)
        if 'LP-HYS' in code:
            hys_code = code
            compatible_codes.add(hys_code)
        if 'LP-CW-SIRM' in code:
            sirm_code = code
            compatible_codes.add(sirm_code)

    measurements_df = measurements_df[measurements_df['method_codes'].isin(compatible_codes)].copy()

    # Find the correct temperature column
    temperature_data = None

    def get_column_if_valid(measurements_df, column_name) -> pd.Series|None:
        if column_name not in measurements_df:
            return None
        
        data = measurements_df[column_name]

        data = pd.to_numeric(data, errors="coerce")
        if data.isna().any():
            return None
            
        # Check that there is variation
        if data.min() == data.max():
            return None
        
        return data

    # Get Temperature Column
    temperature_data = get_column_if_valid(measurements_df, 'meas_temp') # Default to `meas_temp`
    if temperature_data is None:
        temperature_data = get_column_if_valid(measurements_df, 'treat_temp')
    if temperature_data is None:
        raise ValueError("Cannot find appropriate Temperature column.")
    
    measurements_df['Temperature (K)'] = temperature_data
    if 'magn_moment' in measurements_df:
        measurements_df['Moment (emu)'] = measurements_df['magn_moment'] * 1000
    if 'magn_mass' in measurements_df:
        measurements_df['Moment (A⋅m2/kg)'] = measurements_df['magn_mass']

    if 'meas_field_dc' in measurements_df:
        measurements_df['Magnetic Field (Oe)'] = measurements_df['meas_field_dc'] / 0.0001

    if 'specimen' in measurements_df:
        experiment_column = 'specimen'
    elif 'experiment' in measurements_df:
        experiment_column = 'experiment'
    else:
        raise ValueError(f"Cannot find `specimen` column or `experiment` column in data")
    
    experiment_names = measurements_df[experiment_column].unique()
    for experiment in experiment_names:
        print(f"Experiment/Specimen: {experiment}")
        specimen_measurements_df = measurements_df[measurements_df[experiment_column] == experiment]

        if experiment_column not in specimens_df:
            experiment_column += "s"
        assert experiment_column in specimens_df, f"Cannot find `{experiment_column}` in {specimens_df.columns}"
        specimen_data = specimens_df[specimens_df[experiment_column] == experiment]
        
        mass = specimen_data["weight"].values[0] * 1_000_000 if 'weight' in specimen_data and len(specimen_data) else 0

        
        write_measurement(
            output_dir, 
            experiment, 
            "RTSIRM", 
            mass, 
            specimen_measurements_df[specimen_measurements_df['method_codes'] == sirm_code],
        )
        write_measurement(
            output_dir, 
            experiment, 
            "ZFCFC", 
            mass, 
            pd.concat([
                specimen_measurements_df[specimen_measurements_df['method_codes'] == zfc_code],
                specimen_measurements_df[specimen_measurements_df['method_codes'] == fc_code]
            ])
        )
        if hys_code:
            write_measurement(
                output_dir, 
                experiment, 
                "HYST", 
                mass, 
                specimen_measurements_df[specimen_measurements_df['method_codes'] == hys_code],
                x_column="Magnetic Field (Oe)",
            )
    
return generated_files

if __name__ == "__main__":
    app()