import numpy as np
from pathlib import Path
from .data import Hysteresis, RTSIRM, ZFCFC, IronOxide, collate_results, data_files_list, iron_oxides_list

def get_variable_names(iron_oxides:list[IronOxide]) -> list[str]:
    return [f"{iron_oxide}_proportion" for iron_oxide in iron_oxides] + ["total_iron_oxide_proportion", "sigma"]


def build_model(observed:np.ndarray, basis_functions:list[np.ndarray], iron_oxides:list[IronOxide]) -> np.ndarray:
    import pymc as pm
    
    # Number of basis functions
    k = len(basis_functions)

    assert len(iron_oxides) == k

    # Stack the basis functions into a (n_observations x k) matrix
    X = np.column_stack(basis_functions)

    # Alpha parameter for the Dirichlet distribution
    alpha = np.ones(k)  # Uniform prior, can be modified to reflect different beliefs

    # Create the PyMC model
    with pm.Model() as model:
        # Define the Dirichlet prior for the proportions
        iron_oxide_proportions = pm.Dirichlet("iron_oxide_proportions", a=alpha)

        # Give names to the proportions
        for i, iron_oxide in enumerate(iron_oxides):
            pm.Deterministic(f"{iron_oxide}_proportion", iron_oxide_proportions[i])

        total_iron_oxide_proportion = pm.Beta("total_iron_oxide_proportion", alpha=1, beta=1)

        # Define the linear combination of the basis functions
        linear_combination = pm.math.dot(X, iron_oxide_proportions) / total_iron_oxide_proportion

        # Likelihood: Assume the observations are normally distributed around the linear combination
        sigma = pm.HalfCauchy("sigma", beta=1)
        pm.Normal("likelihood", mu=linear_combination, sigma=sigma, observed=observed)

    return model


def sample_posterior(model, samples:int=1_000):
    import pymc as pm

    with model:
        # Sample from the posterior
        inference_data = pm.sample(samples, return_inferencedata=True)
    
    return inference_data


def run_inference(
    hysteresis_path: Path|None,
    rtsirm_path: Path|None,
    zfcfc_path: Path|None,
    iron_oxides: list[IronOxide],
    samples: int = 1_000,
) -> np.ndarray:
    data_files = data_files_list(hysteresis_path, rtsirm_path, zfcfc_path)

    # collate results
    observed, basis_functions = collate_results(data_files, iron_oxides)

    model = build_model(observed, basis_functions, iron_oxides)
    inference_data = sample_posterior(model, samples=samples)
    return inference_data
