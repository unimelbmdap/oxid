import numpy as np

def build_model(observed:np.ndarray, basis_functions:list[np.ndarray]) -> np.ndarray:
    import pymc as pm
    
    # Number of basis functions
    k = len(basis_functions)

    # Stack the basis functions into a (n_observations x k) matrix
    X = np.column_stack(basis_functions)

    # Alpha parameter for the Dirichlet distribution
    alpha = np.ones(k)  # Uniform prior, can be modified to reflect different beliefs

    # Create the PyMC model
    with pm.Model() as model:
        # Define the Dirichlet prior for the proportions
        proportions = pm.Dirichlet("proportions", a=alpha)

        fe_proportion = pm.Beta("fe_proportion", alpha=1, beta=1)

        # Define the linear combination of the basis functions
        # The last basis function contributes nothing but is included in the sum
        linear_combination = pm.math.dot(X, proportions) / fe_proportion

        # Likelihood: Assume the observations are normally distributed around the linear combination
        sigma = pm.HalfCauchy("sigma", beta=1)
        pm.Normal("likelihood", mu=linear_combination, sigma=sigma, observed=observed)

    return model


def sample_posterior(model, samples:int=1_000):
    import pymc as pm

    with model:
        # Sample from the posterior
        trace = pm.sample(samples, return_inferencedata=True)
    
    return trace
