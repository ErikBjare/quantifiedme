import click

from .location import locate
from .habitbull import habits
from .activitywatch import activitywatch
from .qslang import main as qslang
from .oura import oura


@click.group()
def quantifiedme():
    pass


for subcmd in [locate, habits, oura, activitywatch, qslang]:
    quantifiedme.add_command(subcmd)


if __name__ == "__main__":
    quantifiedme()
