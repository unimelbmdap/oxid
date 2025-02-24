import numpy as np
from pathlib import Path
from rich.console import Console
import pytensor.tensor as pt
from patsy import dmatrix

from .data import IronOxide, collate_results, data_files_list

console = Console()

TUNE_DEFAULT=1_000
DRAWS_DEFAULT=2_000
CHAINS_DEFAULT=4
CORES_DEFAULT=4


def get_variable_names(iron_oxides:list[IronOxide]) -> list[str]:
    return [f"{iron_oxide}_factor" for iron_oxide in iron_oxides] + ["sigma_observations"]


def build_model_basis_functions(observations: list[np.ndarray], basis_functions_list: list[list[np.ndarray]], regimes:list[str], iron_oxides: list[IronOxide], num_knots=4) -> "pm.Model":
    import pymc as pm
    import numpy as np

    assert len(observations) == len(basis_functions_list) == len(regimes)

    k = len(iron_oxides)

    alpha = np.ones(k)
    
    with pm.Model() as model:
        # Proportions of iron oxides
        iron_oxide_proportions = pm.Dirichlet("iron_oxide_proportions", a=alpha)
        for i, iron_oxide in enumerate(iron_oxides):
            pm.Deterministic(f"{iron_oxide}_proportion", iron_oxide_proportions[i])

        total_iron_oxide_proportion = pm.Beta("total_iron_oxide_proportion", alpha=1, beta=1)

        # Residual independent noise
        sigma_observations = pm.HalfNormal("sigma_observations", sigma=0.01)

        for observed, basis_functions, regime in zip(observations, basis_functions_list, regimes):
            assert len(basis_functions) == k
            X = np.column_stack(basis_functions)            

            # Linear combination of basis functions
            linear_combination = pm.Deterministic(f"linear_combination_{regime}", pm.math.dot(X, iron_oxide_proportions) / total_iron_oxide_proportion)


            x_np = np.arange(len(observed))
            knots = np.linspace(0, len(observed), num_knots)

            # Create spline basis using Patsy's dmatrix
            B = dmatrix(
                "bs(year, knots=knots, degree=2, include_intercept=True) - 1",
                {"year": x_np, "knots": knots[1:-1]},
            )
            B_tensor = pt.constant(B)

            # Define weights for regression
            w = pm.Normal(f"warping_w_{regime}", sigma=0.1, mu=0, shape=B.shape[1])

            # Compute spline regression using dot product
            warping = pm.Deterministic(f"warping_{regime}", pm.math.dot(B_tensor, w.T))
            # warping = 0

            predicted = pm.Deterministic(f"predicted_{regime}", linear_combination + warping)

            # Likelihood
            pm.Normal(f"likelihood_{regime}", mu=predicted, sigma=sigma_observations, observed=observed)

    return model


def build_model_rescale(observations: list[np.ndarray], basis_functions_list: list[list[np.ndarray]], regimes:list[str], iron_oxides: list[IronOxide], num_knots=4) -> "pm.Model":
    import pymc as pm
    import numpy as np

    assert len(observations) == len(basis_functions_list) == len(regimes)

    k = len(iron_oxides)
    
    with pm.Model() as model:
        # Proportions of iron oxides
        factors = pm.Beta("factors", alpha=1.0, beta=1.0, shape=k)
        for i, iron_oxide in enumerate(iron_oxides):
            pm.Deterministic(f"{iron_oxide}_factor", factors[i])

        # Residual independent noise
        sigma_observations = pm.HalfNormal("sigma_observations", sigma=0.01)

        for observed, basis_functions, regime in zip(observations, basis_functions_list, regimes):
            assert len(basis_functions) == k
            

            def rescale(array):
                return array / array.max()
                return (array - array.min())/(array.max()-array.min())    

            observed = rescale(observed)
            X = np.column_stack([rescale(basis_function) for basis_function in basis_functions])

            # Linear combination of basis functions
            linear_combination = pm.Deterministic(f"linear_combination_{regime}", pm.math.dot(X, factors))

            x_np = np.arange(len(observed))
            knots = np.linspace(0, len(observed), num_knots)

            # Create spline basis using Patsy's dmatrix
            B = dmatrix(
                "bs(year, knots=knots, degree=2, include_intercept=True) - 1",
                {"year": x_np, "knots": knots[1:-1]},
            )
            B_tensor = pt.constant(B)

            # Define weights for regression
            # w = pm.Normal(f"warping_w_{regime}", sigma=0.1, mu=0, shape=B.shape[1])

            # Compute spline regression using dot product
            # warping = pm.Deterministic(f"warping_{regime}", pm.math.dot(B_tensor, w.T))
            warping = 0

            predicted = pm.Deterministic(f"predicted_{regime}", linear_combination + warping)

            # Likelihood
            pm.Normal(f"likelihood_{regime}", mu=predicted, sigma=sigma_observations, observed=observed)

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

# build_model = build_model_basis_functions
build_model = build_model_rescale

def sample_posterior(model, draws:int=DRAWS_DEFAULT, tune:int=TUNE_DEFAULT, chains:int=CHAINS_DEFAULT, cores:int=CORES_DEFAULT) -> "pm.InferenceData":
    import pymc as pm

    with model:
        # Sample from the posterior
        inference_data = pm.sample(draws=draws, tune=tune, return_inferencedata=True, cores=cores, chains=chains)
    
    return inference_data


def posterior_predictive_check(model, inference_data, regimes:list[str]) -> np.ndarray:
    import pymc as pm

    with model:
        ppc = pm.sample_posterior_predictive(inference_data)

    ppc_dict = {"posterior_predictive": ppc["posterior_predictive"]}

    # for dataset in regimes:
    #     for regime, (start, end) in regimes[dataset].items():
    #         ppc_dict[f"linear_combination_{dataset}_{regime}"] = inference_data.posterior["predicted"].isel(predicted_dim_0=slice(start, end))
    #         ppc_dict[f"posterior_predictive_{dataset}_{regime}"] = ppc.posterior_predictive["likelihood"].isel(likelihood_dim_2=slice(start, end))
    #         ppc_dict[f"observed_{dataset}_{regime}"] = inference_data["observed_data"]["likelihood"].isel(likelihood_dim_0=slice(start, end))

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
    chains: int = CHAINS_DEFAULT,
    cores: int = CORES_DEFAULT,
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

    model = build_model(observed, basis_functions, regimes, iron_oxides)
    inference_data = sample_posterior(model, draws=draws, tune=tune, chains=chains, cores=cores)
    posterior_predictive_check(model, inference_data, regimes)

    return inference_data
