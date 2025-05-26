import typer
from pathlib import Path
import pandas as pd

import warnings
warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6",
    category=FutureWarning,
    module="sklearn"
)


from .data import Hysteresis, RTSIRM, ZFCFC, collate_results, data_files_list, iron_oxides_list
from .viz import plot_moment, plot_components, plot_strip
from .viz import plot_standards as plot_standards_viz
from .viz import plot_inputs as plot_inputs_viz
from .features import dimensionality_reduction, build_feature_vectors

app = typer.Typer(pretty_exceptions_enable=False)


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
def embed(
    csvs:list[Path] = typer.Argument(help="Path to the file with the paths"),
    hysteresis:bool=typer.Option(True, help="Whether to use hysteresis data"),
    rtsirm:bool=typer.Option(True, help="Whether to use RT-SIRM data"),
    zfcfc:bool=typer.Option(True, help="Whether to use ZFC-FC data"),
    points:int=typer.Option(250, help="Number of points to interpolate to"),
    features:int=typer.Option(20, help="Number of features to extract. If zero then it uses the raw data"),
    n_neighbors:int=typer.Option(15, help="Number of neighbors for UMAP"),
    reducer:Path=typer.Option(None, help="Path to save the UMAP reducer file"),
    image:Path=typer.Option(None, help="Path to save the image"),
    output_csv:Path=typer.Option(None, help="Path to save the output CSV"),
    title:str=typer.Option("Embeddings", help="Title of the plot"),
    seed:int=typer.Option(0, help="Random seed for UMAP"),
    include_normalized:bool=typer.Option(True, help="Whether to include normalized data"),
    include_unnormalized:bool=typer.Option(True, help="Whether to include unnormalized data"),
    show:bool=typer.Option(True, help="Whether to show the plot"),
    base_dir:Path=typer.Option(None, help="Base directory for the data files. If not provided, it will use the directory of the CSV file"),
    components:int=typer.Option(2, help="Number of components for UMAP"),
    color:str=typer.Option("Group", help="Column name to use for coloring the points"),
    force:bool=typer.Option(False, help="Force re-computation of the reducer if it already exists"),
):
    """ Embed the data from a CSV file using UMAP and plot the results. """

    assert len(csvs) >= 1
    dfs = []
    for csv in csvs:
        df = pd.read_csv(csv)
        if 'base_dir' not in df:
            my_base_dir = base_dir or csv.parent.resolve()
            df['base_dir'] = str(my_base_dir)

        dfs.append(df)
    df = pd.concat( dfs )
    
    vectors = build_feature_vectors(
        df,
        hysteresis=hysteresis,
        rtsirm=rtsirm,
        zfcfc=zfcfc,
        points=points,
        features=features,
        include_normalized=include_normalized,
        include_unnormalized=include_unnormalized,
    )

    transformed_data = dimensionality_reduction(
        vectors,
        n_neighbors=n_neighbors,
        seed=seed,
        reducer_path=reducer,
        n_components=components,
        force=force,
    )

    # Plot the results
    if show or image:
        if components == 2:
            plot_components(
                transformed_data,
                df,
                title=title,
                output=image,
                show=show,
                color_column=color,
            )
        else:
            plot_strip(
                transformed_data,
                df,
                title=title,
                output=image,
                show=show,
                color_column=color,
            )

    # Save CSV
    if output_csv:
        # Save the transformed data to a CSV file
        transformed_df = pd.DataFrame(transformed_data, columns=[f"Component_{i+1}" for i in range(transformed_data.shape[1])])

        df = pd.concat([df, transformed_df], axis=1)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved CSV to {output_csv}")
        