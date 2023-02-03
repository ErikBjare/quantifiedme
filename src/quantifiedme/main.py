import click

from .load.location import locate
from .load.habitbull import habits
from .derived.screentime import screentime
from .load.qslang import main as qslang
from .load.oura import oura


@click.group()
def main():
    """QuantifiedMe is a tool to help you track your life"""
    pass


for subcmd in [locate, habits, oura, screentime]:
    main.add_command(subcmd)
main.add_command(qslang, name="qslang")


if __name__ == "__main__":
    main()
