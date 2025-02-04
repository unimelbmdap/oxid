import typer
from pathlib import Path
import pandas as pd
import arviz as az

from .data import Hysteresis, RTSIRM, ZFCFC, collate_results, data_files_list, iron_oxides_list
from .viz import plot_moment
from .viz import plot_standards as plot_standards_viz
from .viz import plot_inputs as plot_inputs_viz
from .viz import plot_posterior_histograms
from .viz import plot_posterior_predictive_check as plot_posterior_predictive_check_viz
from .models import get_variable_names, run_inference, DRAWS_DEFAULT, TUNE_DEFAULT

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
    tune:int=typer.Option(TUNE_DEFAULT, help="Number of samples to tune the sampler"),
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
        draws=draws,
        tune=tune,
        gradients=gradients,
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

    def noramlize_column_name(name):
        return name.lower().replace("-", "")

    def find_column(df, name):
        name = noramlize_column_name(name)
        for column in df.columns:
            if name in noramlize_column_name(column):
                return column
        return None
    
    hysteresis_column = find_column(df, "hysteresis")
    rtsirm_column = find_column(df, "rtsirm")
    zfcfc_column = find_column(df, "zfcfc")

    variable_names = get_variable_names(iron_oxides)

    base_dir = csv.parent.resolve()
    def get_path(column) -> Path|None:
        return base_dir/row[column] if column and row[column] else None
        
    for i, row in df.iterrows():
        inference_data = run_inference(
            hysteresis_path=get_path(hysteresis_column) if hysteresis else None,
            rtsirm_path=get_path(rtsirm_column) if rtsirm else None,
            zfcfc_path=get_path(zfcfc_column) if zfcfc else None,
            iron_oxides=iron_oxides,
            draws=draws,
            tune=tune,
            gradients=gradients,
        )
    
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
    mode:str=typer.Option('markers', help="Plot mode: 'markers' or 'lines+markers' or 'lines'"),
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
    observed, basis_functions, _ = collate_results(data_files, iron_oxides, gradients=gradients)

    plot_inputs_viz(observed, basis_functions, iron_oxides, rescale=rescale, show=show, output=output, mode=mode)


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