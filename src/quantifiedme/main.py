import click

from .derived.all_df import all_df
from .derived.screentime import screentime
from .derived.sleep import sleep
from .load.habitbull import habits
from .load.location import locate
from .load.oura import oura
from .load.qslang import main as qslang


@click.group()
def main():
    """QuantifiedMe is a tool to help you track your life"""
    pass


for subcmd in [locate, habits, oura, screentime, all_df, sleep]:
    main.add_command(subcmd)
main.add_command(qslang, name="qslang")


if __name__ == "__main__":
    main()
