# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import h5py

from dotenv import find_dotenv, load_dotenv
from . import kursiv, lorenz63, lorenz96


def store_data_to_h5(f, name, t, x):
    logger = logging.getLogger(__name__)
    logger.info(f"Storing {name} to hdf5")
    grp = f.create_group(name)
    grp.create_dataset('t', data=t)
    grp.create_dataset('x', data=x)


@click.command()
@click.argument('output_filepath', type=click.Path())
@click.option('-n', '--n-init', type=int)
def main(output_filepath, n_init):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Opening hdf5 file at {output_filepath}")
    with h5py.File(output_filepath, "w") as f:

        for key, run in [
            ('kuramoto', kursiv.run),
            ('lorenz96', lorenz96.run),
            ('lorenz63', lorenz63.run),
        ]:
            logger.info(f"Running {key} simulations")
            t, x = run(n_init)
            store_data_to_h5(f, key, t, x)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
