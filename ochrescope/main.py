import typer
from pathlib import Path

# from .data import read_data
from .viz import plot_moment
from .viz import plot_standards as plot_standards_viz

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

    data = []
    standards = []


@app.command()
def plot(
    file:Path = typer.Argument(help="Path to a data file"),
):
    data = read_data(file)
    plot_moment(data).show()


@app.command()
def plot_standards():
    plot_standards_viz().show()
