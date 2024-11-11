import typer
from pathlib import Path
import pandas as pd
import arviz as az

# from .data import read_data
from .data import Hysteresis, RTSIRM, ZFCFC, IronOxide, collate_results, data_files_list, iron_oxides_list
from .viz import plot_moment
from .viz import plot_standards as plot_standards_viz
from .viz import plot_inputs as plot_inputs_viz
from .models import build_model, sample_posterior, get_variable_names, run_inference

app = typer.Typer()


@app.command()
def infer(
    hysteresis: Path = typer.Option(None, help="Path to a hysteresis data file"),
    rtsirm: Path = typer.Option(None, help="Path to a RT-SIRM data file"),
    zfcfc: Path = typer.Option(None, help="Path to a ZFC-FC data file"),
    magnetite:bool=typer.Option(True, help="Whether the infer the proportion of magnetite in the sample"),
    hematite:bool=typer.Option(True, help="Whether the infer the proportion of hematite in the sample"),
    goethite:bool=typer.Option(True, help="Whether the infer the proportion of goethite in the sample"),
    samples:int=typer.Option(1_000, help="Number of samples to draw from the posterior"),
    plot:Path=typer.Option(None, help="Path to save the posterior plot"),
    inference_data:Path=typer.Option(None, help="Path to save the inference data"),
):
    typer.echo("Analyzing...")
    inference_data_path = Path(inference_data) if inference_data else None

    # Create list of Iron Oxide Types to use
    iron_oxides = iron_oxides_list(magnetite, hematite, goethite)

    inference_data = run_inference(
        hysteresis_path=hysteresis,
        rtsirm_path=rtsirm,
        zfcfc_path=zfcfc,
        iron_oxides=iron_oxides,
        samples=samples,
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
    csv: Path = typer.Option(..., help="Path of the CSV file. Needs to have columns 'Hysteresis', 'RT-SIRM', 'ZFC-FC'"),
    output: Path = typer.Option(None, help="Path to save the output CSV"),
    inplace:bool=typer.Option(False, help="Whether to save the output CSV in place"),
    magnetite:bool=typer.Option(True, help="Whether the infer the proportion of magnetite in the sample"),
    hematite:bool=typer.Option(True, help="Whether the infer the proportion of hematite in the sample"),
    goethite:bool=typer.Option(True, help="Whether the infer the proportion of goethite in the sample"),
    samples:int=typer.Option(1_000, help="Number of samples to draw from the posterior"),
):
    df = pd.read_csv(csv)

    if output is None and not inplace:  
        raise ValueError("Either --output or --inplace must be provided")
    
    if inplace:
        output = csv

    # Create list of Iron Oxide Types to use
    iron_oxides = iron_oxides_list(magnetite, hematite, goethite)

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

    for i, row in df.iterrows():
        typer.echo("Analyzing...")
        inference_data = run_inference(
            hysteresis_path=row[hysteresis_column] if hysteresis_column else None,
            rtsirm_path=row[rtsirm_column] if rtsirm_column else None,
            zfcfc_path=row[zfcfc_column] if zfcfc_column else None,
            iron_oxides=iron_oxides,
            samples=samples,
        )
    
        # Print summary
        summary = az.summary(inference_data)
        summary = summary[summary.index.isin(variable_names)]

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
):
    typer.echo("Analyzing...")

    # Create list of data files
    data_files = data_files_list(hysteresis, rtsirm, zfcfc)

    # Create list of Iron Oxide Types to use
    iron_oxides = iron_oxides_list(magnetite, hematite, goethite)

    # collate results
    observed, basis_functions = collate_results(data_files, iron_oxides)

    plot_inputs_viz(observed, basis_functions).show()


@app.command()
def plot_rtsirm(
    file:Path = typer.Argument(help="Path to a data file"),
):
    data = RTSIRM(file)
    plot_moment(data, title=file.name).show()


@app.command()
def plot_standards():
    plot_standards_viz().show()
