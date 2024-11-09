import pandas as pd

def read_data(file_path, standardize:bool=True) -> pd.DataFrame:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_start_index = next(i for i, line in enumerate(lines) if '[Data]' in line)

    df = pd.read_csv(file_path, skiprows=data_start_index + 1)

    if df['Moment (emu)'].isnull().values.any():
        df['Moment (emu)'] = df['DC Moment Fixed Ctr (emu)']
    
    if standardize:
        mass_line = lines[23].strip() # hack, should find the line with SAMPLE_MASS
        components = mass_line.split(",")
        assert components[2] == "SAMPLE_MASS"
        mass = float(components[1])
        df['Moment (emu)'] = df['Moment (emu)'] / mass * 1000

    return df


