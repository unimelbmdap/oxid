import numpy as np
from pathlib import Path
from rich.console import Console

from .data import IronOxide, collate_results, data_files_list

console = Console()

TUNE_DEFAULT=1_000
DRAWS_DEFAULT=1_000


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
        linear_combination = pm.Deterministic("linear_combination", pm.math.dot(X, iron_oxide_proportions) / total_iron_oxide_proportion)

        # Likelihood: Assume the observations are normally distributed around the linear combination
        sigma = pm.HalfCauchy("sigma", beta=1)
        pm.Normal("likelihood", mu=linear_combination, sigma=sigma, observed=observed)

    return model


def sample_posterior(model, draws:int=DRAWS_DEFAULT, tune:int=TUNE_DEFAULT):
    import pymc as pm

    with model:
        # Sample from the posterior
        inference_data = pm.sample(draws=draws, tune=tune, return_inferencedata=True)
    
    return inference_data


def posterior_predictive_check(model, inference_data, regimes) -> np.ndarray:
    import pymc as pm

    with model:
        ppc = pm.sample_posterior_predictive(inference_data)

    ppc_dict = {"posterior_predictive": ppc["posterior_predictive"]}

    for dataset in regimes:
        for regime, (start, end) in regimes[dataset].items():
            ppc_dict[f"linear_combination_{dataset}_{regime}"] = inference_data.posterior["linear_combination"].isel(linear_combination_dim_0=slice(start, end))
            ppc_dict[f"posterior_predictive_{dataset}_{regime}"] = ppc.posterior_predictive["likelihood"].isel(likelihood_dim_2=slice(start, end))
            ppc_dict[f"observed_{dataset}_{regime}"] = inference_data["observed_data"]["likelihood"].isel(likelihood_dim_0=slice(start, end))

    # Add posterior predictive samples to the inference data
    inference_data.add_groups(**ppc_dict)

    return ppc

def run_inference(
    hysteresis_path: Path|None,
    rtsirm_path: Path|None,
    zfcfc_path: Path|None,
    iron_oxides: list[IronOxide],
    draws: int = DRAWS_DEFAULT,
    tune: int = TUNE_DEFAULT,
) -> np.ndarray:
    console.rule("OxID Inference")
    console.print("[bold green]Inferring iron oxide proportions from:")
    if hysteresis_path and hysteresis_path.exists():
        console.print(f"  [red]Hysteresis:[/red] {hysteresis_path}")
    if rtsirm_path and rtsirm_path.exists():
        console.print(f"  [red]RT-SIRM:[/red] {rtsirm_path}")
    if zfcfc_path and zfcfc_path.exists():
        console.print(f"  [red]ZFC-FC:[/red] {zfcfc_path}")

    data_files = data_files_list(hysteresis_path, rtsirm_path, zfcfc_path)

    # collate results
    observed, basis_functions, regimes = collate_results(data_files, iron_oxides)

    model = build_model(observed, basis_functions, iron_oxides)
    inference_data = sample_posterior(model, draws=draws, tune=tune)
    posterior_predictive_check(model, inference_data, regimes)

    return inference_data
