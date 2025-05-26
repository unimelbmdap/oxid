from pathlib import Path
import umap
import numpy as np
from collections import defaultdict
import pandas as pd

from .data import Hysteresis, RTSIRM, ZFCFC

def noramlize_column_name(name):
    return name.lower().replace("-", "")


def find_column(df, name):
    name = noramlize_column_name(name)
    for column in df.columns:
        if name in noramlize_column_name(column):
            return column
    return None


def build_feature_vectors(
    df: pd.DataFrame,
    base_dir: Path,
    hysteresis: bool = True,
    rtsirm: bool = True,
    zfcfc: bool = True,
    features: int = None,
    include_normalized: bool = True,
    include_unnormalized: bool = False,
    points: int = 250, # Number of points to interpolate to
) -> np.ndarray:
    hysteresis_column = find_column(df, "hysteresis") if hysteresis else None
    rtsirm_column = find_column(df, "rtsirm") if rtsirm else None
    zfcfc_column = find_column(df, "zfcfc") if zfcfc else None

    results = defaultdict(dict)
    
    min_temp = 15
    max_temp = 297
    field_extreme = 69500
    min_values = {'Cooling': min_temp, 'Heating': min_temp, 'ZFC': min_temp, 'FC': min_temp, 'Decreasing': -field_extreme, 'Increasing': field_extreme}
    max_values = {'Cooling': max_temp, 'Heating': max_temp, 'ZFC': max_temp, 'FC': max_temp, 'Decreasing': -field_extreme, 'Increasing': field_extreme}
    x_values = dict()
    for regime in min_values:
        x_values[regime] = np.linspace(min_values[regime], max_values[regime], points)

    vectors = []
    for i, row in df.iterrows():
        results[i].update(row)

        def get_path(column) -> Path|None:
            return base_dir/row[column] if column and row[column] else None

        datasets = []
        if hysteresis_column:
            datasets.append(Hysteresis(get_path(hysteresis_column)))
        if zfcfc_column:
            datasets.append(ZFCFC(get_path(zfcfc_column)))
        if rtsirm_column:
            datasets.append(RTSIRM(get_path(rtsirm_column)))

        feature_vectors = []
        for dataset in datasets:
            print(f"extracting {dataset.path}")
            data = dataset.extract()

            # Get the maximum value for this 
            max_value = None
            for regime, arrays in data.items():
                my_max = arrays[1].max()
                max_value = my_max if max_value is None else np.maximum(max_value, my_max)

            for regime, arrays in data.items():
                # Interpolate results to grid
                x,y = arrays[0], arrays[1]
                interpolated = np.interp(x_values[regime], x, y)

                # Extract features
                if features:
                    fft_values = np.fft.fft(interpolated)
                    positive_magnitudes = np.abs(fft_values[:len(fft_values)//2])
                    feature_vector = positive_magnitudes[:features]
                else:
                    feature_vector = interpolated

                assert include_normalized or include_unnormalized, f"You must include at least one of normalized or unnormalized data"
                if include_normalized:
                    feature_vectors.append(feature_vector/max_value)
                if include_unnormalized:
                    feature_vectors.append(feature_vector)

        feature_vector = np.concatenate(feature_vectors)
        vectors.append(feature_vector)
    
    vectors = np.asarray(vectors)
    return vectors    


def dimensionality_reduction(
    vectors: np.ndarray,
    n_neighbors:int = 15,
    seed:int = 0,
    n_components:int = 2,
    reducer_path: Path|str|None = None,
    force:bool = False,
):
    """ 
    Perform dimensionality reduction on the input vectors using UMAP.
    If a reducer_path is provided and the file exists, it will load the UMAP model from the file.
    Otherwise, it will fit a new UMAP model and save it to the specified path.
    """
    if reducer_path and Path(reducer_path).exists() and not force:
        import pickle
        with open(reducer_path, "rb") as f:
            model = pickle.load(f)
    else:
        model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=seed)
        model.fit(vectors)
        if reducer_path:
            import pickle
            Path(reducer_path).parent.mkdir(exist_ok=True, parents=True)
            print(f"Writing UMAP reducer to {reducer_path}")
            with open(reducer_path, "wb") as f:
                pickle.dump(model, f)
    
    # Perform the transformation
    return model.transform(vectors)
    