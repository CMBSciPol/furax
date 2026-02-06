import typer

from .prepare import app as prepare_app
from .run import app as run_app

app = typer.Typer(help='Multi-observation mapmaking for SO data.')
app.add_typer(prepare_app)
app.add_typer(run_app)


if __name__ == '__main__':
    app()
