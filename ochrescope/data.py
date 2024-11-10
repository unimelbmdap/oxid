import pandas as pd
from pathlib import Path
from enum import Enum

STANDARDS_DIR = Path(__file__).parent / "standards"


class Measurement(Enum):
    ZFCFC = 'zfcfc'
    RTSIRM = 'rtsirm'
    HYSTERESIS = 'hysteresis'

    @classmethod
    def get(cls, query:str) -> "Measurement":
        if isinstance(query, cls):
            return query
        return cls(str(query).lower())

    def __str__(self) -> str:
        return self.value


class IronOxide(Enum):
    GOETHITE = 'goethite'
    HEMATITE = 'hematite'
    MAGNETITE = 'magnetite'

    def get_dir(self) -> Path:
        return STANDARDS_DIR / self.value
    
    def get_file(self, measurement:Measurement|str) -> Path:
        return self.get_dir() / f"{self.value}-{measurement}.dat"
    
    @classmethod
    def get(cls, query:str) -> "IronOxide":
        if isinstance(query, cls):
            return query
        return cls(str(query).lower())

    def __str__(self) -> str:
        return self.value
    
    def read_data(self, measurement:Measurement) -> pd.DataFrame:
        return read_data(self.get_file(measurement))


def read_data(file_path:Path|str) -> pd.DataFrame:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_start_index = next(i for i, line in enumerate(lines) if '[Data]' in line)

    df = pd.read_csv(file_path, skiprows=data_start_index + 1)

    if df['Moment (emu)'].isnull().values.any():
        df['Moment (emu)'] = df['DC Moment Fixed Ctr (emu)']
    
    mass_line = lines[23].strip() # hack, should find the line with SAMPLE_MASS
    components = mass_line.split(",")
    assert components[2] == "SAMPLE_MASS"
    mass = float(components[1])
    df['Moment_Am2_per_kg'] = df['Moment (emu)'] / mass * 1000

    return df


def read_standard(iron_oxide:IronOxide|str, measurement:Measurement|str) -> pd.DataFrame:
    iron_oxide = IronOxide.get(iron_oxide)
    return read_data(iron_oxide.get_file(measurement))
