import click

from .location import locate
from .habitbull import habits
from .activitywatch import activitywatch
from .oura import oura


@click.group()
def main():
    pass


main.add_command(locate)
main.add_command(habits)
main.add_command(oura)
main.add_command(activitywatch)


if __name__ == '__main__':
    main()
