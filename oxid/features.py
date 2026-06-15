from pathlib import Path
import umap
import numpy as np
from collections import defaultdict
import pandas as pd
from rich.progress import track

from data import Hysteresis, RTSIRM, ZFCFC

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
    hysteresis: bool = True,
    rtsirm: bool = True,
    zfcfc: bool = True,
    features: int = 20,
    include_normalized: bool = True,
    include_unnormalized: bool = False,
    points: int = 250,
    verbose: bool = False,
) -> np.ndarray:

    hysteresis_column = find_column(df, "hysteresis") if hysteresis else None
    rtsirm_column = find_column(df, "rtsirm") if rtsirm else None
    zfcfc_column = find_column(df, "zfcfc") if zfcfc else None

    min_temp, max_temp = 15, 297
    field_extreme = 69500

    x_values = {
        "Cooling": np.linspace(min_temp, max_temp, points),
        "Heating": np.linspace(min_temp, max_temp, points),
        "ZFC": np.linspace(min_temp, max_temp, points),
        "FC": np.linspace(min_temp, max_temp, points),
        "Decreasing": np.linspace(-field_extreme, field_extreme, points),
        "Increasing": np.linspace(-field_extreme, field_extreme, points),
    }

    vectors = []

    for _, row in track(df.iterrows(), total=len(df), description="Building feature vectors"):

        def get_path(column):
            if not column or not row[column]:
                return None
            path = Path(row["base_dir"]) / str(row[column])
            return path if path.exists() else None

        datasets = []

        if hysteresis and hysteresis_column:
            p = get_path(hysteresis_column)
            if p:
                datasets.append(Hysteresis(p))

        if rtsirm and rtsirm_column:
            p = get_path(rtsirm_column)
            if p:
                datasets.append(RTSIRM(p))

        if zfcfc and zfcfc_column:
            p = get_path(zfcfc_column)
            if p:
                datasets.append(ZFCFC(p))

        if len(datasets) == 0:
            continue

        sample_features = []

        for dataset in datasets:
            data = dataset.extract()

            max_value = None

            # compute scaling
            for regime, arrays in data.items():
                if len(arrays[1]) == 0:
                    continue
                m = np.max(arrays[1])
                max_value = m if max_value is None else max(max_value, m)

            # per regime features
            for regime, arrays in data.items():
                x, y = arrays[0], arrays[1]

                if len(x) == 0 or len(y) == 0:
                    continue

                interp = np.interp(x_values[regime], x, y)

                fft_vals = np.fft.fft(interp)
                mags = np.abs(fft_vals[: len(fft_vals)//2])

                feat = mags[:features]
                feat = np.pad(feat, (0, features - len(feat)))

                if include_normalized:
                    sample_features.append(feat / max_value if max_value else feat)

                if include_unnormalized:
                    sample_features.append(feat)

        if len(sample_features) == 0:
            continue

        feature_vector = np.concatenate(sample_features)

        if verbose:
            print(row.get("Name", ""), feature_vector.shape)

        vectors.append(feature_vector)

    # enforce consistency
    lengths = {len(v) for v in vectors}

    print("Vector lengths:", sorted(lengths))

    if len(lengths) != 1:
        raise ValueError(f"Inconsistent feature vector lengths: {sorted(lengths)}")

    return np.asarray(vectors)

def dimensionality_reduction(
    vectors: np.ndarray,
    n_neighbors:int = 15,
    min_dist:float = 0.1,
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
        model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=seed, min_dist=min_dist)
        model.fit(vectors)
        if reducer_path:
            import pickle
            Path(reducer_path).parent.mkdir(exist_ok=True, parents=True)
            print(f"Writing UMAP reducer to {reducer_path}")
            with open(reducer_path, "wb") as f:
                pickle.dump(model, f)
    
    # Perform the transformation
    result = np.concatenate([model.transform(vectors[i:i+1]) for i in range(len(vectors))])
    return result
    