import pandas as pd
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from functools import cached_property
import numpy as np

STANDARDS_DIR = Path(__file__).parent / "standards"

@dataclass
class Data():
    path:Path|str

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        with open(self.path, 'r') as file:
            lines = file.readlines()

        data_start_index = next(i for i, line in enumerate(lines) if '[Data]' in line)

        df = pd.read_csv(self.path, skiprows=data_start_index + 1)

        if df['Moment (emu)'].isnull().values.any():
            df['Moment (emu)'] = df['DC Moment Fixed Ctr (emu)']
        
        mass_line = lines[23].strip() # hack, should find the line with SAMPLE_MASS
        components = mass_line.split(",")
        assert components[2] == "SAMPLE_MASS"
        mass = float(components[1])
        df['Moment (A⋅m2/kg)'] = df['Moment (emu)'] / mass * 1000

        return df
    
    def extract(self) -> dict[str, tuple[np.ndarray,np.ndarray]]:
        raise NotImplementedError

    @property
    def x_axis(self) -> str:
        raise NotImplementedError


class Hysteresis(Data):
    @property
    def x_axis(self) -> str:
        return 'Magnetic Field (Oe)'

    def extract(self) -> dict[str, tuple[np.ndarray,np.ndarray]]:
        df = self.dataframe
        decreasing = df[self.x_axis].diff().fillna(0) > 0
        decreasing_df = df[decreasing]
        increasing_df = df[~decreasing]

        return dict(
            Decreasing=(decreasing_df[self.x_axis].values, decreasing_df['Moment (A⋅m2/kg)'].values),
            Increasing=(increasing_df[self.x_axis].values, increasing_df['Moment (A⋅m2/kg)'].values),
        )
    
    def color(self, regime:str) -> str:
        match(regime):
            case 'Decreasing':
                return 'purple'
            case 'Increasing':
                return 'orange'
        return 'black'


class RTSIRM(Data):
    @property
    def x_axis(self) -> str:
        return 'Temperature (K)'

    def extract(self) -> dict[str, tuple[np.ndarray,np.ndarray]]:
        df = self.dataframe
        temp_diff_positive = df[self.x_axis].diff().fillna(0) > 0
        heating_df = df[temp_diff_positive]
        cooling_df = df[~temp_diff_positive]

        return dict(
            Cooling=(cooling_df[self.x_axis].values, cooling_df['Moment (A⋅m2/kg)'].values),
            Heating=(heating_df[self.x_axis].values, heating_df['Moment (A⋅m2/kg)'].values),
        )
    
    def color(self, regime:str) -> str:
        match(regime):
            case 'Cooling':
                return 'blue'
            case 'Heating':
                return 'red'
        return 'black'


class ZFCFC(Data):
    @property
    def x_axis(self) -> str:
        return 'Temperature (K)'

    def extract(self) -> dict[str, tuple[np.ndarray,np.ndarray]]:
        df = self.dataframe
        temp_diff = df["Temperature (K)"].diff()
        transition_index = temp_diff[temp_diff < 0].index[0]

        # Label ZFC and FC based on the transition index
        df["Regime"] = ["ZFC" if i < transition_index else "FC" for i in df.index]

        zfc_data = df[df["Regime"] == "ZFC"]
        fc_data = df[df["Regime"] == "FC"]

        return dict(
            ZFC=(zfc_data[self.x_axis].values, zfc_data['Moment (A⋅m2/kg)'].values),
            FC=(fc_data[self.x_axis].values, fc_data['Moment (A⋅m2/kg)'].values),
        )    
    
    def color(self, regime:str) -> str:
        match(regime):
            case 'ZFC':
                return 'green'
            case 'FC':
                return 'yellow'
        return 'black'


DATA_TYPES = {
    'hysteresis': Hysteresis,
    'rtsirm': RTSIRM,
    'zfcfc': ZFCFC,
}


class IronOxide(Enum):
    GOETHITE = 'goethite'
    HEMATITE = 'hematite'
    MAGNETITE = 'magnetite'

    def get_dir(self) -> Path:
        return STANDARDS_DIR / self.value
    
    def get_file(self, data_type:str) -> Path:
        return self.get_dir() / f"{self.value}-{data_type}.dat"
    
    @classmethod
    def get(cls, query:str) -> "IronOxide":
        if isinstance(query, cls):
            return query
        return cls(str(query).lower())

    def __str__(self) -> str:
        return self.value
    
    def standard_data(self, data_type:str) -> pd.DataFrame:
        return DATA_TYPES[data_type](self.get_file(data_type))


def standard_data(iron_oxide:IronOxide|str, measurement:str) -> pd.DataFrame:
    iron_oxide = IronOxide.get(iron_oxide)
    return iron_oxide.standard_data(measurement)
