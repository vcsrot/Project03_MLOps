import pandas as pd
import pytest


@pytest.fixture(scope='session')
def data():

    # Check if cleaned version is on path:
    data_path = 'starter/data/census_clean.csv'

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path)

    return df