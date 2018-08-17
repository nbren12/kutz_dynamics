import click


@click.command()
@click.argument('model', type=click.Path())
@click.argument('data', type=click.Path())
@click.argument('problem', type=str)
@click.argument('output_path', type=click.Path())
@click.option('-b', '--burn-in', type=int)
def main(data, model, problem, output_path, burn_in):
    print(locals())
    pass


if __name__ == '__main__':
    main()
