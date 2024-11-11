import typer
from pathlib import Path
import numpy as np
import arviz as az

# from .data import read_data
from .data import Hysteresis, RTSIRM, ZFCFC, IronOxide, collate_results, data_files_list, iron_oxides_list
from .viz import plot_moment
from .viz import plot_standards as plot_standards_viz
from .viz import plot_inputs as plot_inputs_viz
from .models import build_model, sample_posterior

app = typer.Typer()


@app.command()
def analyze(
    hysteresis: Path = typer.Option(None, help="Path to a hysteresis data file"),
    rtsirm: Path = typer.Option(None, help="Path to a RT-SIRM data file"),
    zfcfc: Path = typer.Option(None, help="Path to a ZFC-FC data file"),
    magnetite:bool=typer.Option(True, help="Whether the infer the proportion of magnetite in the sample"),
    hematite:bool=typer.Option(True, help="Whether the infer the proportion of hematite in the sample"),
    goethite:bool=typer.Option(True, help="Whether the infer the proportion of goethite in the sample"),
    samples:int=typer.Option(1_000, help="Number of samples to draw from the posterior"),
):
    typer.echo("Analyzing...")

    # Create list of data files
    data_files = data_files_list(hysteresis, rtsirm, zfcfc)

    # Create list of Iron Oxide Types to use
    iron_oxides = iron_oxides_list(magnetite, hematite, goethite)

    # collate results
    observed, basis_functions = collate_results(data_files, iron_oxides)

    model = build_model(observed, basis_functions)
    trace = sample_posterior(model, samples=samples)
    print(az.summary(trace))
    breakpoint()
    

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
