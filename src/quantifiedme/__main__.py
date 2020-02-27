import click

from .location import locate
from .habitbull import habits


@click.group()
def main():
    pass


main.add_command(locate)
main.add_command(habits)


if __name__ == '__main__':
    main()
