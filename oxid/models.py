import numpy as np
from pathlib import Path
from rich.console import Console

from .data import IronOxide, collate_results, data_files_list

console = Console()

TUNE_DEFAULT=1_000
DRAWS_DEFAULT=1_000


def get_variable_names(iron_oxides:list[IronOxide]) -> list[str]:
    return [f"{iron_oxide}_proportion" for iron_oxide in iron_oxides] + ["total_iron_oxide_proportion", "sigma"]


def build_model_basis_functions(observed: np.ndarray, basis_functions: list[np.ndarray], iron_oxides: list[IronOxide]) -> "pm.Model":
    import pymc as pm
    import numpy as np

    x_coords = np.arange(len(observed))

    k = len(basis_functions)
    assert len(iron_oxides) == k

    X = np.column_stack(basis_functions)
    alpha = np.ones(k)

    with pm.Model() as model:
        # Proportions of iron oxides
        iron_oxide_proportions = pm.Dirichlet("iron_oxide_proportions", a=alpha)
        for i, iron_oxide in enumerate(iron_oxides):
            pm.Deterministic(f"{iron_oxide}_proportion", iron_oxide_proportions[i])

        total_iron_oxide_proportion = pm.Beta("total_iron_oxide_proportion", alpha=1, beta=1)

        # Linear combination of basis functions
        linear_combination = pm.Deterministic("linear_combination", pm.math.dot(X, iron_oxide_proportions) / total_iron_oxide_proportion)

        # Gaussian Process for smooth noise
        length_scale = pm.Gamma("length_scale", alpha=2, beta=1)  # Smoothness
        amplitude = pm.HalfNormal("amplitude", sigma=1)  # Scale of f

        cov_func = amplitude**2 * pm.gp.cov.ExpQuad(1, length_scale)
        gp_noise = pm.gp.Latent(cov_func=cov_func)
        f = gp_noise.prior("gp_noise", X=x_coords.reshape(-1, 1))

        predicted_mu = pm.Deterministic("predicted_mu", linear_combination + f)

        # Residual independent noise
        sigma_obs = pm.HalfCauchy("sigma_obs", beta=1)

        # Likelihood
        pm.Normal("likelihood", mu=predicted_mu, sigma=sigma_obs, observed=observed)

    return model


def build_model_other(observed:np.ndarray, basis_functions:list[np.ndarray], iron_oxides:list[IronOxide]):
    import pymc as pm
    
    # Number of basis functions
    k = len(basis_functions)

    assert len(iron_oxides) == k

    # Stack the basis functions into a (n_observations x k) matrix
    X = np.column_stack(basis_functions)

    # Alpha parameter for the Dirichlet distribution
    alpha = np.ones(k + 1)  # Uniform prior, can be modified to reflect different beliefs

    # Create the PyMC model
    with pm.Model() as model:
        # Define the Dirichlet prior for the proportions
        iron_oxide_proportions = pm.Dirichlet("iron_oxide_proportions", a=alpha)

        # Give names to the proportions
        for i, iron_oxide in enumerate(iron_oxides):
            pm.Deterministic(f"{iron_oxide}_proportion", iron_oxide_proportions[i])

        # Define the linear combination of the basis functions
        linear_combination = pm.Deterministic("linear_combination", pm.math.dot(X, iron_oxide_proportions[:k]))

        # Likelihood: Assume the observations are normally distributed around the linear combination
        sigma = pm.HalfCauchy("sigma", beta=1)
        pm.Normal("likelihood", mu=linear_combination, sigma=sigma, observed=observed)

    return model

build_model = build_model_basis_functions

def sample_posterior(model, draws:int=DRAWS_DEFAULT, tune:int=TUNE_DEFAULT):
    import pymc as pm

    with model:
        # Sample from the posterior
        inference_data = pm.sample(draws=draws, tune=tune, return_inferencedata=True, cores=1, chains=2)
    
    return inference_data


def posterior_predictive_check(model, inference_data, regimes) -> np.ndarray:
    import pymc as pm

    with model:
        ppc = pm.sample_posterior_predictive(inference_data)

    ppc_dict = {"posterior_predictive": ppc["posterior_predictive"]}

    for dataset in regimes:
        for regime, (start, end) in regimes[dataset].items():
            ppc_dict[f"linear_combination_{dataset}_{regime}"] = inference_data.posterior["predicted_mu"].isel(predicted_mu_dim_0=slice(start, end))
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
    gradients: bool = False,
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
    observed, basis_functions, regimes = collate_results(data_files, iron_oxides, gradients=gradients)

    model = build_model(observed, basis_functions, iron_oxides)
    inference_data = sample_posterior(model, draws=draws, tune=tune)
    posterior_predictive_check(model, inference_data, regimes)

    return inference_data
