import typer
from pathlib import Path
import pandas as pd
import arviz as az

# from .data import read_data
from .data import Hysteresis, RTSIRM, ZFCFC, IronOxide, collate_results, data_files_list, iron_oxides_list
from .viz import plot_moment
from .viz import plot_standards as plot_standards_viz
from .viz import plot_inputs as plot_inputs_viz
from .models import build_model, sample_posterior, get_variable_names, run_inference, DRAWS_DEFAULT, TUNE_DEFAULT

app = typer.Typer()


@app.command()
def infer(
    hysteresis: Path = typer.Option(None, help="Path to a hysteresis data file"),
    rtsirm: Path = typer.Option(None, help="Path to a RT-SIRM data file"),
    zfcfc: Path = typer.Option(None, help="Path to a ZFC-FC data file"),
    magnetite:bool=typer.Option(True, help="Whether the infer the proportion of magnetite in the sample"),
    hematite:bool=typer.Option(True, help="Whether the infer the proportion of hematite in the sample"),
    goethite:bool=typer.Option(True, help="Whether the infer the proportion of goethite in the sample"),
    maghemite:bool=typer.Option(True, help="Whether the infer the proportion of maghemite in the sample"),
    draws:int=typer.Option(DRAWS_DEFAULT, help="Number of samples to draw from the posterior"),
    tune:int=typer.Option(TUNE_DEFAULT, help="Number of samples to tune the sampler"),
    plot:Path=typer.Option(None, help="Path to save the posterior plot"),
    inference_data:Path=typer.Option(None, help="Path to save the inference data"),
):
    typer.echo("Analyzing...")
    inference_data_path = Path(inference_data) if inference_data else None

    # Create list of Iron Oxide Types to use
    iron_oxides = iron_oxides_list(goethite, hematite, magnetite, maghemite)

    inference_data = run_inference(
        hysteresis_path=hysteresis,
        rtsirm_path=rtsirm,
        zfcfc_path=zfcfc,
        iron_oxides=iron_oxides,
        draws=draws,
        tune=tune,
    )
    
    # Print summary
    variable_names = get_variable_names(iron_oxides)
    summary = az.summary(inference_data)
    summary = summary[summary.index.isin(variable_names)]
    print(summary)

    if inference_data_path:
        inference_data_path.parent.mkdir(parents=True, exist_ok=True)
        inference_data.to_netcdf(inference_data_path)

    if plot:
        import matplotlib.pyplot as plt
        plot = Path(plot)
        plot.parent.mkdir(parents=True, exist_ok=True)
        
        az.plot_posterior(
            inference_data,
            var_names=variable_names,
        )
        plt.savefig(plot)
    

@app.command()
def infer_csv(
    csv: Path = typer.Argument(help="Path of the CSV file. Needs to have columns 'Hysteresis', 'RT-SIRM', 'ZFC-FC'"),
    output: Path = typer.Option(None, help="Path to save the output CSV"),
    inplace:bool=typer.Option(False, help="Whether to save the output CSV in place"),
    magnetite:bool=typer.Option(True, help="Whether the infer the proportion of magnetite in the sample"),
    hematite:bool=typer.Option(True, help="Whether the infer the proportion of hematite in the sample"),
    goethite:bool=typer.Option(True, help="Whether the infer the proportion of goethite in the sample"),
    maghemite:bool=typer.Option(True, help="Whether the infer the proportion of maghemite in the sample"),
    draws:int=typer.Option(DRAWS_DEFAULT, help="Number of samples to draw from the posterior"),
    tune:int=typer.Option(TUNE_DEFAULT, help="Number of samples to tune the sampler"),
):
    df = pd.read_csv(csv)

    if output is None and not inplace:  
        raise ValueError("Either --output or --inplace must be provided")
    
    if inplace:
        output = csv

    # Create list of Iron Oxide Types to use
    iron_oxides = iron_oxides_list(goethite, hematite, magnetite, maghemite)

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
            hysteresis_path=get_path(hysteresis_column),
            rtsirm_path=get_path(rtsirm_column),
            zfcfc_path=get_path(zfcfc_column),
            iron_oxides=iron_oxides,
            draws=draws,
            tune=tune,
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
    magnetite:bool=typer.Option(True, help="Whether the infer the proportion of magnetite in the sample"),
    hematite:bool=typer.Option(True, help="Whether the infer the proportion of hematite in the sample"),
    goethite:bool=typer.Option(True, help="Whether the infer the proportion of goethite in the sample"),
    maghemite:bool=typer.Option(True, help="Whether the infer the proportion of maghemite in the sample"),
):
    # Create list of data files
    data_files = data_files_list(hysteresis, rtsirm, zfcfc)

    # Create list of Iron Oxide Types to use
    iron_oxides = iron_oxides_list(goethite, hematite, magnetite, maghemite)

    # collate results
    observed, basis_functions = collate_results(data_files, iron_oxides)

    plot_inputs_viz(observed, basis_functions).show()


@app.command()
def plot_rtsirm(
    file:Path = typer.Argument(help="Path to a data file"),
    show:bool = typer.Option(True, help="Whether to show the plot"),
    output:Path = typer.Option(None, help="Path to save the plot"),
):
    data = RTSIRM(file)
    plot_moment(data, title=file.name, show=show, output=output)


@app.command()
def plot_zfcfc(
    file:Path = typer.Argument(help="Path to a data file"),
    show:bool = typer.Option(True, help="Whether to show the plot"),
    output:Path = typer.Option(None, help="Path to save the plot"),
):
    data = ZFCFC(file)
    plot_moment(data, title=file.name, show=show, output=output)


@app.command()
def plot_hysteresis(
    file:Path = typer.Argument(help="Path to a data file"),
    show:bool = typer.Option(True, help="Whether to show the plot"),
    output:Path = typer.Option(None, help="Path to save the plot"),
):
    data = Hysteresis(file)
    plot_moment(data, title=file.name, show=show, output=output)


@app.command()
def plot_standards(
    show:bool = typer.Option(True, help="Whether to show the plot"),
    output:Path = typer.Option(None, help="Path to save the plot"),        
):
    plot_standards_viz(show=show, output=output)
