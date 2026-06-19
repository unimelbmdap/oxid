import re
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
        print(df.columns.tolist())
        
        # find line with 'SAMPLE_MASS,0.123'
        mass = None
        for line in lines:
            if line.strip().endswith('SAMPLE_MASS') and line.startswith('INFO,'):
                components = line.split(",")
                mass = float(
                    re.sub(r'[^0-9eE.+-]', '', components[1])
                )
        assert mass is not None, "Could not find SAMPLE_MASS in data file"

        # Calculate 'Moment (A⋅m2/kg)' if not present
        if 'Moment (A⋅m2/kg)' not in df:
            if 'Moment..emu.' in df:
                df['Moment (emu)'] = df['Moment..emu.']

            moment_column = 'Moment (emu)'

        if df[moment_column].isnull().values.any():

            if 'DC Moment Fixed Ctr (emu)' not in df.columns:
                st.error("Missing column: DC Moment Fixed Ctr (emu)")
                st.write("Available columns:")
                st.write(df.columns.tolist())

                # stop here so you can inspect the columns
                raise ValueError(
                    f"Missing column 'DC Moment Fixed Ctr (emu)'. "
                    f"Available columns: {df.columns.tolist()}"
                )

            df[moment_column] = df['DC Moment Fixed Ctr (emu)']

            if not pd.api.types.is_numeric_dtype(df[moment_column]):
                df[moment_column] = df[moment_column].astype(str).str.replace(r'[^0-9.eE-]', '', regex=True).astype(float)
            df['Moment (A⋅m2/kg)'] = df[moment_column] / mass * 1000
        return df
    
    def extract(self) -> dict[str, tuple[np.ndarray,np.ndarray]]:
        raise NotImplementedError

    @property
    def x_axis(self) -> str:
        raise NotImplementedError
    
    def interpolate_standards(self, iron_oxides:list["IronOxide"], npoints:int=0) -> np.ndarray:
        my_extracted = self.extract()
        standards = [IronOxide.get(iron_oxide).standard_data(self.__class__.__name__.lower()).extract() for iron_oxide in iron_oxides]

        # Truncate my data to the same range as the standard data
        for standard_extracted in standards:
            assert my_extracted.keys() == standard_extracted.keys()
            for key in my_extracted.keys():
                x, y = my_extracted[key]
                standard_x, _ = standard_extracted[key]

                # make sure the x values are sorted
                sorted_indices = np.argsort(x)
                x = x[sorted_indices]
                y = y[sorted_indices]

                # Test if the x values are in the same range, if not, then truncate my data
                if x[0] < standard_x.min() or x[-1] > standard_x.max():
                    y = y[(x >= standard_x.min()) & (x <= standard_x.max())]
                    x = x[(x >= standard_x.min()) & (x <= standard_x.max())]  

                my_extracted[key] = (x, y)

        # Interpolate standard data to my x values
        result = {}
        for key in my_extracted.keys():
            x, y = my_extracted[key]

            if npoints:
                new_x = np.linspace(x.min(), x.max(), npoints)
                y = np.interp(new_x, x, y)
                x = new_x

            interpolated_array = []
            for standard_extracted in standards:
                standard_x, standard_y = standard_extracted[key]

                # make sure the x values are sorted
                sorted_indices = np.argsort(standard_x)
                standard_x = standard_x[sorted_indices]
                standard_y = standard_y[sorted_indices]

                interpolated_array.append(np.interp(x, standard_x, standard_y))

            result[key] = (x, y, interpolated_array)
        
        return result


class Hysteresis(Data):
    @property
    def x_axis(self) -> str:
        return 'Magnetic Field (Oe)'

    def extract(self) -> dict[str, tuple[np.ndarray,np.ndarray]]:
        df = self.dataframe

        # Rename improperly named column
        df = df.rename(columns={'Magnetic.Field..Oe.': self.x_axis})

        if not pd.api.types.is_numeric_dtype(df[self.x_axis]):
            df[self.x_axis] = df[self.x_axis].astype(str).str.replace(r'[^0-9.eE-]', '', regex=True).astype(float)

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

    @classmethod
    def title(cls) -> str:
        return "Hysteresis"


class RTSIRM(Data):
    @property
    def x_axis(self) -> str:
        return 'Temperature (K)'

    def extract(self) -> dict[str, tuple[np.ndarray,np.ndarray]]:
        df = self.dataframe

        # Rename improperly named column
        df = df.rename(columns={'Temperature..K.': self.x_axis})

        if not pd.api.types.is_numeric_dtype(df[self.x_axis]):
            df[self.x_axis] = df[self.x_axis].astype(str).str.replace(r'[^0-9.eE-]', '', regex=True).astype(float)

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

    @classmethod
    def title(cls) -> str:
        return "RT-SIRM"


class ZFCFC(Data):
    @property
    def x_axis(self) -> str:
        return 'Temperature (K)'

    def extract(self) -> dict[str, tuple[np.ndarray,np.ndarray]]:
        df = self.dataframe

        # Rename improperly named column
        df = df.rename(columns={'Temperature..K.': self.x_axis})

        if not pd.api.types.is_numeric_dtype(df["Temperature (K)"]):
            df["Temperature (K)"] = df["Temperature (K)"].astype(str).str.replace(r'[^0-9.eE-]', '', regex=True).astype(float)

        temp_diff = df["Temperature (K)"].diff()
        try:
            transition_index = temp_diff[temp_diff < 0].index[0]
        except IndexError:
            transition_index = len(df) - 1

        # Label ZFC and FC based on the transition index
        df["Regime"] = ["ZFC" if i < transition_index else "FC" for i in df.index]

        zfc_data = df[df["Regime"] == "ZFC"]
        fc_data = df[df["Regime"] == "FC"]

        # remove first point for both ZFC and FC
        zfc_data = zfc_data.iloc[1:]
        fc_data = fc_data.iloc[1:]

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

    @classmethod
    def title(cls) -> str:
        return "ZFC-FC"


DATA_TYPES = {
    'hysteresis': Hysteresis,
    'rtsirm': RTSIRM,
    'zfcfc': ZFCFC,
}


class IronOxide(Enum):
    GOETHITE = 'goethite'
    HEMATITE = 'hematite'
    MAGNETITE = 'magnetite'
    MAGHEMITE = 'maghemite'
    ALGOETHITE = 'algoethite'

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
    
    def standard_data(self, data_type:str) -> Data:
        return DATA_TYPES[data_type](self.get_file(data_type))
    
    def title(self):
        if self == IronOxide.ALGOETHITE:
            return "Al-Goethite"
        return self.value.title()
    
    @property
    def color(self) -> str:
        match self:
            case IronOxide.GOETHITE:
                return 'cyan'
            case IronOxide.HEMATITE:
                return 'red'
            case IronOxide.MAGNETITE:
                return 'purple'
            case IronOxide.MAGHEMITE:
                return 'orange'
            case IronOxide.ALGOETHITE:
                return 'blue'
        return 'grey'


def standard_data(iron_oxide:IronOxide|str, measurement:str) -> Data:
    iron_oxide = IronOxide.get(iron_oxide)
    return iron_oxide.standard_data(measurement)


def collate_results(data_files:list[Data], iron_oxides:list[IronOxide], gradients:bool=False) -> tuple[list[np.ndarray], list[list[np.ndarray]], list[str], list[str]]:
    observations = []
    basis_functions = []
    regimes = []
    datatypes = []
    for data in data_files: 
        datatype = type(data).__name__
        result = data.interpolate_standards(iron_oxides)
        for regime, value in result.items():
            _, y, standards = value
            datatypes.append(datatype)
            
            if gradients:
                y = np.diff(y)

            observations.append(y)
            regimes.append(regime)

            regime_basis_functions = []
            for iron_oxide_index in range(len(iron_oxides)):
                basis_function = standards[iron_oxide_index]
                if gradients:
                    basis_function = np.diff(basis_function)

                regime_basis_functions.append(basis_function)
            basis_functions.append(regime_basis_functions)

    return observations, basis_functions, regimes, datatypes


def data_files_list(
    hysteresis: Path = None,
    rtsirm: Path = None,
    zfcfc: Path = None,
) -> list[Data]:
    data_files = []
    if hysteresis:
        data_files.append(Hysteresis(hysteresis))
    if rtsirm:
        data_files.append(RTSIRM(rtsirm))
    if zfcfc:
        data_files.append(ZFCFC(zfcfc))

    return data_files


def iron_oxides_list(goethite:bool, hematite:bool, magnetite:bool, maghemite:bool, algoethite:bool) -> list[IronOxide]:
    iron_oxides = []
    if magnetite:
        iron_oxides.append(IronOxide.MAGNETITE)
    if hematite:
        iron_oxides.append(IronOxide.HEMATITE)
    if goethite:
        iron_oxides.append(IronOxide.GOETHITE)
    if maghemite:
        iron_oxides.append(IronOxide.MAGHEMITE)
    if algoethite:
        iron_oxides.append(IronOxide.ALGOETHITE)
    return iron_oxides