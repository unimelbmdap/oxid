import typer
from pathlib import Path

app = typer.Typer()


@app.command()
def analyze(path: Path):
    typer.echo("Analyzing ochres...")