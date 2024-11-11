import typer
from pathlib import Path
import numpy as np
import arviz as az

# from .data import read_data
from .data import Hysteresis, RTSIRM, ZFCFC, IronOxide
from .viz import plot_moment
from .viz import plot_standards as plot_standards_viz
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
):
    typer.echo("Analyzing...")

    # Create list of data files
    data_files = []
    if hysteresis:
        data_files.append(Hysteresis(hysteresis))
    if rtsirm:
        data_files.append(RTSIRM(rtsirm))
    if zfcfc:
        data_files.append(ZFCFC(zfcfc))

    # Create list of Iron Oxide Types to use
    iron_oxides = []
    if magnetite:
        iron_oxides.append(IronOxide.MAGNETITE)
    if hematite:
        iron_oxides.append(IronOxide.HEMATITE)
    if goethite:
        iron_oxides.append(IronOxide.GOETHITE)


    # collate results
    observed = np.empty((0,))
    basis_functions = [np.empty((0,))] * len(iron_oxides)
    for data in data_files: 
        result = data.interpolate_standards(iron_oxides)
        for _, value in result.items():
            _, y, standards = value
            observed = np.concatenate((observed, y))
            for iron_oxide_index in range(len(iron_oxides)):
                basis_functions[iron_oxide_index] = np.concatenate((basis_functions[iron_oxide_index], standards[iron_oxide_index]))


    model = build_model(observed, basis_functions)
    trace = sample_posterior(model)
    print(az.summary(trace))
    breakpoint()
    

@app.command()
def plot(
    file:Path = typer.Argument(help="Path to a data file"),
):
    data = read_data(file)
    plot_moment(data).show()


@app.command()
def plot_standards():
    plot_standards_viz().show()
