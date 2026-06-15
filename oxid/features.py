from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import umap

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
    features: int = None,
    include_normalized: bool = True,
    include_unnormalized: bool = False,
    points: int = 250,
    verbose: bool = False,
) -> np.ndarray:

    hysteresis_column = (
        find_column(df, "hysteresis")
        if hysteresis else None
    )

    rtsirm_column = (
        find_column(df, "rtsirm")
        if rtsirm else None
    )

    zfcfc_column = (
        find_column(df, "zfcfc")
        if zfcfc else None
    )

    results = defaultdict(dict)

    min_temp = 15
    max_temp = 297
    field_extreme = 69500

    min_values = {
        "Cooling": min_temp,
        "Heating": min_temp,
        "ZFC": min_temp,
        "FC": min_temp,
        "Decreasing": -field_extreme,
        "Increasing": field_extreme,
    }

    max_values = {
        "Cooling": max_temp,
        "Heating": max_temp,
        "ZFC": max_temp,
        "FC": max_temp,
        "Decreasing": -field_extreme,
        "Increasing": field_extreme,
    }

    x_values = {}

    for regime in min_values:
        x_values[regime] = np.linspace(
            min_values[regime],
            max_values[regime],
            points,
        )

    vectors = []

    for i, row in track(
        df.iterrows(),
        total=len(df),
        description="Building feature vectors",
    ):

        results[i].update(row)

        def get_path(column):

            if not column:
                return None

            if column not in row:
                return None

            value = row[column]

            if pd.isna(value):
                return None

            path = Path(row["base_dir"]) / str(value)

            if not path.exists():
                return None

            return path

        datasets = []

        if hysteresis_column:
            path = get_path(hysteresis_column)
            if path:
                datasets.append(Hysteresis(path))

        if zfcfc_column:
            path = get_path(zfcfc_column)
            if path:
                datasets.append(ZFCFC(path))

        if rtsirm_column:
            path = get_path(rtsirm_column)
            if path:
                datasets.append(RTSIRM(path))

        # Skip samples with no selected datasets
        if len(datasets) == 0:
            continue

        feature_vectors = []

        for dataset in datasets:

            if verbose:
                print(f"Extracting {dataset.path}")

            try:
                data = dataset.extract()
            except Exception as e:
                print(f"Skipping {dataset.path}: {e}")
                continue

            max_value = None

            for regime, arrays in data.items():

                if len(arrays[1]) == 0:
                    print(
                        f"Skipping empty regime "
                        f"{regime} in {dataset.path}"
                    )
                    continue

                my_max = arrays[1].max()

                if max_value is None:
                    max_value = my_max
                else:
                    max_value = np.maximum(
                        max_value,
                        my_max,
                    )

            if max_value is None or max_value == 0:
                continue

            for regime, arrays in data.items():

                if len(arrays[1]) == 0:
                    continue

                x = arrays[0]
                y = arrays[1]

                interpolated = np.interp(
                    x_values[regime],
                    x,
                    y,
                )

                if features:

                    fft_values = np.fft.fft(
                        interpolated
                    )

                    positive_magnitudes = np.abs(
                        fft_values[
                            : len(fft_values) // 2
                        ]
                    )

                    current_feature = (
                        positive_magnitudes[:features]
                    )

                else:
                    current_feature = interpolated

                if include_normalized:
                    feature_vectors.append(
                        current_feature / max_value
                    )

                if include_unnormalized:
                    feature_vectors.append(
                        current_feature
                    )

        if len(feature_vectors) == 0:
            continue

        feature_vector = np.concatenate(
            feature_vectors
        )

        print(
            row.get("Name", f"row_{i}"),
            [type(d).__name__ for d in datasets],
            len(feature_vector),
        )

        vectors.append(feature_vector)

    if len(vectors) == 0:
        raise ValueError(
            "No valid feature vectors were generated."
        )

    lengths = {len(v) for v in vectors}

    print("Vector lengths:", sorted(lengths))

    if len(lengths) != 1:
        raise ValueError(
            f"Inconsistent feature vector lengths: "
            f"{sorted(lengths)}"
        )

    vectors = np.asarray(vectors)

    return vectors


def dimensionality_reduction(
    vectors: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 0,
    n_components: int = 2,
    reducer_path: Path | str | None = None,
    force: bool = False,
):

    if (
        reducer_path
        and Path(reducer_path).exists()
        and not force
    ):

        import pickle

        with open(reducer_path, "rb") as f:
            model = pickle.load(f)

    else:

        model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            random_state=seed,
            min_dist=min_dist,
        )

        model.fit(vectors)

        if reducer_path:

            import pickle

            Path(reducer_path).parent.mkdir(
                exist_ok=True,
                parents=True,
            )

            print(
                f"Writing UMAP reducer to "
                f"{reducer_path}"
            )

            with open(reducer_path, "wb") as f:
                pickle.dump(model, f)

    result = np.concatenate(
        [
            model.transform(vectors[i:i + 1])
            for i in range(len(vectors))
        ]
    )

    return result