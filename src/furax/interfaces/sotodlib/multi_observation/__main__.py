from cyclopts import App

from .prepare import prepare
from .run import run

app = App(help='Multi-observation mapmaking for SO data.')
app.command(prepare)
app.command(run)


if __name__ == '__main__':
    app()
