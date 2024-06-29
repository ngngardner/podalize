"""Main Podalize cli application module."""

import typer

app = typer.Typer()


@app.command()
def main() -> None:
    """Podalize main command."""
    typer.echo("Hello world")
