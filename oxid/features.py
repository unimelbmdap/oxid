from pathlib import Path
import umap
import numpy as np


def dimensionality_reduction(
    vectors: np.ndarray,
    n_neighbors:int = 15,
    seed:int = 42,
    reducer_path: Path|str|None = None,
):
    """ 
    Perform dimensionality reduction on the input vectors using UMAP.
    If a reducer_path is provided and the file exists, it will load the UMAP model from the file.
    Otherwise, it will fit a new UMAP model and save it to the specified path.
    """
    if reducer_path and Path(reducer_path).exists():
        import pickle
        with open(reducer_path, "rb") as f:
            model = pickle.load(f)
    else:
        model = umap.UMAP(n_neighbors=n_neighbors, n_components=2, random_state=seed)
        model.fit(vectors)
        if reducer_path:
            import pickle
            Path(reducer_path).parent.mkdir(exist_ok=True, parents=True)
            print(f"Writing UMAP reducer to {reducer_path}")
            with open(reducer_path, "wb") as f:
                pickle.dump(model, f)
    
    # Perform the transformation
    return model.transform(vectors)
    