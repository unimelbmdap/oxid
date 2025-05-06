import typer
from pathlib import Path
import pandas as pd
import numpy as np
import arviz as az
from collections import defaultdict

from .data import Hysteresis, RTSIRM, ZFCFC, collate_results, data_files_list, iron_oxides_list
from .viz import plot_moment, plot_components, plot_strip
from .viz import plot_standards as plot_standards_viz
from .viz import plot_inputs as plot_inputs_viz
from .viz import plot_posterior_histograms
from .viz import plot_posterior_predictive_check as plot_posterior_predictive_check_viz
from .models import get_variable_names, run_inference, DRAWS_DEFAULT, TUNE_DEFAULT, CHAINS_DEFAULT, CORES_DEFAULT
from .features import dimensionality_reduction, find_column, build_feature_vectors
app = typer.Typer()




@app.command()
def infer(
    hysteresis: Path = typer.Option(None, help="Path to a hysteresis data file"),
    rtsirm: Path = typer.Option(None, help="Path to a RT-SIRM data file"),
    zfcfc: Path = typer.Option(None, help="Path to a ZFC-FC data file"),
    magnetite:bool=typer.Option(True, help="Whether to infer the proportion of magnetite in the sample"),
    hematite:bool=typer.Option(True, help="Whether to infer the proportion of hematite in the sample"),
    goethite:bool=typer.Option(True, help="Whether to infer the proportion of goethite in the sample"),
    maghemite:bool=typer.Option(True, help="Whether to infer the proportion of maghemite in the sample"),
    algoethite:bool=typer.Option(True, help="Whether to infer the proportion of al-goethite in the sample"),
    draws:int=typer.Option(DRAWS_DEFAULT, help="Number of samples to draw from the posterior"),
    chains:int=typer.Option(CHAINS_DEFAULT, help="Number of chains to run"),
    tune:int=typer.Option(TUNE_DEFAULT, help="Number of samples to tune the sampler"),
    cores:int=typer.Option(CORES_DEFAULT, help="Number of cores to use"),
    inference_data:Path=typer.Option(None, help="Path to save the inference data"),
    show:bool=typer.Option(True, help="Whether to show the plot"),
    plot:Path=typer.Option(None, help="Path to save the posterior plot"),
    ppc:Path=typer.Option(None, help="Path to save the posterior predictive check plot"),
    gradients:bool=typer.Option(False, help="Whether to use gradients"),
):
    """
    Infer the proportions of iron oxides in a sample using hysteresis, RT-SIRM, and/or ZFC-FC data.
    """
    typer.echo("Analyzing...")
    inference_data_path = Path(inference_data) if inference_data else None

    # Create list of Iron Oxide Types to use
    iron_oxides = iron_oxides_list(goethite, hematite, magnetite, maghemite, algoethite)

    inference_data = run_inference(
        hysteresis_path=hysteresis,
        rtsirm_path=rtsirm,
        zfcfc_path=zfcfc,
        iron_oxides=iron_oxides,
        gradients=gradients,
        chains=chains,
        tune=tune,
        draws=draws,
        cores=cores,
    )
    
    # Print summary
    variable_names = get_variable_names(iron_oxides)
    summary = az.summary(inference_data)
    summary = summary[summary.index.isin(variable_names)]
    print(summary)

    if inference_data_path:
        inference_data_path.parent.mkdir(parents=True, exist_ok=True)
        inference_data.to_netcdf(inference_data_path)

    if plot or show or ppc:
        plot_posterior_histograms(inference_data, show=show, output=plot)
        plot_posterior_predictive_check_viz(inference_data, show=show, output=ppc)        
    

@app.command()
def infer_csv(
    csv: Path = typer.Argument(help="Path of the CSV file. Needs to have columns 'Hysteresis', 'RT-SIRM', 'ZFC-FC'"),
    output: Path = typer.Option(None, help="Path to save the output CSV"),
    inplace:bool=typer.Option(False, help="Whether to save the output CSV in place"),
    magnetite:bool=typer.Option(True, help="Whether to infer the proportion of magnetite in the sample"),
    hematite:bool=typer.Option(True, help="Whether to infer the proportion of hematite in the sample"),
    goethite:bool=typer.Option(True, help="Whether to infer the proportion of goethite in the sample"),
    maghemite:bool=typer.Option(True, help="Whether to infer the proportion of maghemite in the sample"),
    algoethite:bool=typer.Option(True, help="Whether to infer the proportion of al-goethite in the sample"),
    draws:int=typer.Option(DRAWS_DEFAULT, help="Number of samples to draw from the posterior"),
    chains:int=typer.Option(CHAINS_DEFAULT, help="Number of chains to run"),
    cores:int=typer.Option(CORES_DEFAULT, help="Number of cores to use"),
    inference_data_dir:Path=typer.Option(None, help="Path to a directory to save the inference data"),
    tune:int=typer.Option(TUNE_DEFAULT, help="Number of samples to tune the sampler"),
    hysteresis:bool=typer.Option(True, help="Whether to use hysteresis data"),
    rtsirm:bool=typer.Option(True, help="Whether to use RT-SIRM data"),
    zfcfc:bool=typer.Option(True, help="Whether to use ZFC-FC data"),
    gradients:bool=typer.Option(False, help="Whether to use gradients"),
):
    """
    Infer the proportions of iron oxides for multiple samples using hysteresis, RT-SIRM, and/or ZFC-FC data files listed in a CSV.
    """
    df = pd.read_csv(csv)

    if output is None and not inplace:  
        raise ValueError("Either --output or --inplace must be provided")
    
    if inplace:
        output = csv

    # Create list of Iron Oxide Types to use
    iron_oxides = iron_oxides_list(goethite, hematite, magnetite, maghemite, algoethite)
    
    hysteresis_column = find_column(df, "hysteresis")
    rtsirm_column = find_column(df, "rtsirm")
    zfcfc_column = find_column(df, "zfcfc")

    variable_names = get_variable_names(iron_oxides)

    base_dir = csv.parent.resolve()
        
    for i, row in df.iterrows():
        def get_path(column) -> Path|None:
            return base_dir/row[column] if column and row[column] else None

        hysteresis_path = get_path(hysteresis_column) if hysteresis else None
        rtsirm_path = get_path(rtsirm_column) if rtsirm else None
        zfcfc_path = get_path(zfcfc_column) if zfcfc else None

        if inference_data_dir:
            inference_data_dir = Path(inference_data_dir)
            inference_data_dir.mkdir(parents=True, exist_ok=True)

            path_components = [path.stem for path in [hysteresis_path, rtsirm_path, zfcfc_path] if path]
            filename = "_".join(path_components) + ".nc"       
            inference_data_path = inference_data_dir/filename
            if inference_data_path.exists():
                continue

        inference_data = run_inference(
            hysteresis_path=hysteresis_path,
            rtsirm_path=rtsirm_path,
            zfcfc_path=zfcfc_path,
            iron_oxides=iron_oxides,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            gradients=gradients,
        )

        if inference_data_dir:
            print(f"Writing inference data to {inference_data_path}")     
            inference_data.to_netcdf(inference_data_path)

        # Print summary
        summary = az.summary(inference_data)
        summary = summary[summary.index.isin(variable_names)]
        print(summary)

        # Add to CSV
        for variable_name in variable_names:
            for summary_name in summary.columns:
                col_name = f"{variable_name}_{summary_name}"
                # Add columns to df if first iteration
                if i == 0:
                    df[col_name] = ""
                # Add values to df
                df.at[i, col_name] = summary.loc[variable_name, summary_name]
    
        assert output is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
    

@app.command()
def plot_inputs(
    hysteresis: Path = typer.Option(None, help="Path to a hysteresis data file"),
    rtsirm: Path = typer.Option(None, help="Path to a RT-SIRM data file"),
    zfcfc: Path = typer.Option(None, help="Path to a ZFC-FC data file"),
    magnetite:bool=typer.Option(True, help="Whether to plot the magnetite basis function"),
    hematite:bool=typer.Option(True, help="Whether to plot the hematite basis function"),
    goethite:bool=typer.Option(True, help="Whether to plot the goethite basis function"),
    maghemite:bool=typer.Option(True, help="Whether to plot the maghemite basis function"),
    algoethite:bool=typer.Option(True, help="Whether to plot the al-goethite basis function"),
    rescale:bool=typer.Option(True, help="Whether to rescale the plots by the maximum value"),
    show:bool=typer.Option(True, help="Whether to show the plot"),
    output:Path = typer.Option(None, help="Path to save the plot"),
    mode:str=typer.Option('lines+markers', help="Plot mode: 'markers' or 'lines+markers' or 'lines'"),
    gradients:bool=typer.Option(False, help="Whether to use gradients"),
):
    """
    Plot the observed data and basis functions for a sample using hysteresis, RT-SIRM, and/or ZFC-FC
    """
    # Create list of data files
    data_files = data_files_list(hysteresis, rtsirm, zfcfc)

    # Create list of Iron Oxide Types to use
    iron_oxides = iron_oxides_list(goethite, hematite, magnetite, maghemite, algoethite)

    # collate results
    observed, basis_functions, regimes, datatypes = collate_results(data_files, iron_oxides, gradients=gradients)

    plot_inputs_viz(observed, basis_functions, regimes, datatypes, iron_oxides, rescale=rescale, show=show, output=output, mode=mode)


@app.command()
def plot_rtsirm(
    file:Path = typer.Argument(help="Path to a data file"),
    show:bool = typer.Option(True, help="Whether to show the plot"),
    output:Path = typer.Option(None, help="Path to save the plot"),
):
    """ Plot RT-SIRM data of a sample """
    data = RTSIRM(file)
    plot_moment(data, title=file.name, show=show, output=output)


@app.command()
def plot_zfcfc(
    file:Path = typer.Argument(help="Path to a data file"),
    show:bool = typer.Option(True, help="Whether to show the plot"),
    output:Path = typer.Option(None, help="Path to save the plot"),
):
    """ Plot ZFC-FC data of a sample """
    data = ZFCFC(file)
    plot_moment(data, title=file.name, show=show, output=output)


@app.command()
def plot_hysteresis(
    file:Path = typer.Argument(help="Path to a data file"),
    show:bool = typer.Option(True, help="Whether to show the plot"),
    output:Path = typer.Option(None, help="Path to save the plot"),
):
    """ Plot hysteresis data of a sample """
    data = Hysteresis(file)
    plot_moment(data, title=file.name, show=show, output=output)


@app.command()
def plot_standards(
    show:bool = typer.Option(True, help="Whether to show the plot"),
    output:Path = typer.Option(None, help="Path to save the plot"),        
):
    """ Plot standard Hysteresis, RT-SIRM, and ZFC-FC for each iron oxide """
    plot_standards_viz(show=show, output=output)


@app.command()
def plot_posterior(
    inference_data:Path = typer.Argument(help="Path to the inference data file"),
    output:Path = typer.Option(None, help="Path to save the plot"),
    show:bool = typer.Option(True, help="Whether to show the plot"),
):
    """ Plot the posterior distribution of the iron oxide proportions """
    inference_data = az.from_netcdf(inference_data)
    plot_posterior_histograms(inference_data, show=show, output=output)


@app.command()
def plot_posterior_predictive_check(
    inference_data:Path = typer.Argument(help="Path to the inference data file"),
    output:Path = typer.Option(None, help="Path to save the plot"),
    show:bool = typer.Option(True, help="Whether to show the plot"),
):
    """ Plot the posterior predictive check of the iron oxide proportions """
    inference_data = az.from_netcdf(inference_data)
    plot_posterior_predictive_check_viz(inference_data, show=show, output=output)


@app.command()
def pca_old(
    csv:Path = typer.Argument(help="Path to the file with the paths"),
    hysteresis:bool=typer.Option(True, help="Whether to use hysteresis data"),
    rtsirm:bool=typer.Option(True, help="Whether to use RT-SIRM data"),
    zfcfc:bool=typer.Option(True, help="Whether to use ZFC-FC data"),
    points:int=80,
):
    df = pd.read_csv(csv)
    
    hysteresis_column = find_column(df, "hysteresis")
    rtsirm_column = find_column(df, "rtsirm")
    zfcfc_column = find_column(df, "zfcfc")

    # variable_names = get_variable_names(iron_oxides)

    base_dir = csv.parent.resolve()

    results = defaultdict(dict)
    
    regimes = set()

    min_values = dict()
    max_values = dict()        

    for i, row in df.iterrows():
        def get_path(column) -> Path|None:
            return base_dir/row[column] if column and row[column] else None

        hysteresis_path = get_path(hysteresis_column) if hysteresis else None
        rtsirm_path = get_path(rtsirm_column) if rtsirm else None
        zfcfc_path = get_path(zfcfc_column) if zfcfc else None

        print(rtsirm_path, zfcfc_path)
        datasets = [
            RTSIRM(rtsirm_path),
            ZFCFC(zfcfc_path),
            Hysteresis(hysteresis_path),
        ]

        results[i].update(row)

        for dataset in datasets:
            if not dataset.path: 
                continue

            print(f"extracting {dataset.path}")
            data = dataset.extract()
            regimes.update(data.keys())
            max_value = np.max([arrays[1].max() for arrays in data.values()])
            min_value = np.min([arrays[1].min() for arrays in data.values()])
            # print('min', 'max', min_value, max_value)

            for regime, arrays in data.items():
                min_value = arrays[0].min()
                if regime in min_values:
                    min_values[regime] = max(min_value, min_values[regime])
                else:
                    min_values[regime] = min_value
                max_value = arrays[0].max()
                if regime in max_values:
                    max_values[regime] = min(max_value, max_values[regime])
                else:
                    max_values[regime] = max_value


                # arrays = arrays[0], (arrays[1]-min_value)#/(max_value-min_value)
                arrays = arrays[0], (arrays[1])/arrays[1].max()
                print(regime, min_values[regime], max_values[regime] )
                results[i][regime] = arrays
    
    x_values = dict()
    for regime in regimes:
        x_values[regime] = np.linspace(min_values[regime], max_values[regime], points)

    vectors = []
    for values in results.values():
        to_concatenate = []
        for regime in regimes:
            x, y = values[regime]

            sorted_indices = np.argsort(x)
            x = x[sorted_indices]
            y = y[sorted_indices]

            interpolated = np.interp(x_values[regime], x, y)
            fft_values = np.fft.fft(interpolated)
            positive_magnitudes = np.abs(fft_values[:len(fft_values)//2])
            features = positive_magnitudes[:20]
            # features = interpolated
            # features = interpolated/interpolated.max()

            to_concatenate.append(features)
        vectors.append(np.concatenate(to_concatenate))
    
    vectors = np.asarray(vectors)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(vectors)
    transformed_data = pca.transform(vectors)
    print("Principal Components:")
    print(pca.components_)

    print("\nExplained Variance Ratio:")
    print(pca.explained_variance_ratio_)

    names = [results[i]["RTSIRM"] for i in results]
    color = [results[i]["Cluster"] for i in results]
    import plotly.express as px
    fig = px.scatter(x=transformed_data[:,0], y=transformed_data[:,1], color=color, hover_data=[names])
    fig.update_traces(marker_size=10)
    fig.show()


@app.command()
def pca(
    csv:Path = typer.Argument(help="Path to the file with the paths"),
    hysteresis:bool=typer.Option(True, help="Whether to use hysteresis data"),
    rtsirm:bool=typer.Option(True, help="Whether to use RT-SIRM data"),
    zfcfc:bool=typer.Option(True, help="Whether to use ZFC-FC data"),
    points:int=typer.Option(250, help="Number of points to interpolate to"),
    features:int=typer.Option(20, help="Number of features to extract. If zero then it uses the raw data"),
    n_neighbors:int=typer.Option(15, help="Number of neighbors for UMAP"),
    reducer:Path=typer.Option(None, help="Path to save the UMAP reducer file"),
    image:Path=None,
    title:str="",
    seed:int=typer.Option(0, help="Random seed for UMAP"),
    include_normalized:bool=typer.Option(True, help="Whether to include normalized data"),
    include_unnormalized:bool=typer.Option(True, help="Whether to include unnormalized data"),
    show:bool=typer.Option(True, help="Whether to show the plot"),
    base_dir:Path=typer.Option(None, help="Base directory for the data files. If not provided, it will use the directory of the CSV file"),
    components:int=typer.Option(2, help="Number of components for UMAP"),
    color:str=typer.Option("Group", help="Column name to use for coloring the points"),
):
    df = pd.read_csv(csv)
    base_dir = base_dir or csv.parent.resolve()
    
    vectors = build_feature_vectors(
        df,
        base_dir=base_dir,
        hysteresis=hysteresis,
        rtsirm=rtsirm,
        zfcfc=zfcfc,
        points=points,
        features=features,
        include_normalized=include_normalized,
        include_unnormalized=include_unnormalized,
    )

    # UMAP dimensionality reduction
    transformed_data = dimensionality_reduction(
        vectors,
        n_neighbors=n_neighbors,
        seed=seed,
        reducer_path=reducer,
        n_components=components,
    )

    # Plot the results
    if show or image:
        if components == 2:
            plot_components(
                transformed_data,
                df,
                title=title or "UMAP Projection",
                image=image,
                show=show,
                color_column=color,
            )
        else:
            plot_strip(
                transformed_data,
                df,
                title=title or "UMAP Projection",
                image=image,
                show=show,
                color_column=color,
            )

    # Save CSV
        