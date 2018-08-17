#!/usr/bin/env python
"""A sample command line program for training a model"""

import click


@click.command()
@click.argument('h5file', type=click.Path())
@click.argument('problem', type=str)
@click.argument('output_path', type=click.Path())
@click.option('--lr', help='optional argument', type=float, default=.01)
def main(h5file, problem, output_path, **kwargs):
    print(locals())
    pass


if __name__ == '__main__':
    main()
