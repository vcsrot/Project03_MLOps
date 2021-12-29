import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def test_columns_names(data):
    expected_columns = [
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary"
    ]

    curr_columns = data.columns.values
    assert list(expected_columns) == list(curr_columns), \
        logger.info("Wrong column names.")


def test_valid_education_range(data, min_level=1, max_level=16):
    idx = data['education-num'].between(min_level, max_level)
    assert np.sum(~idx) == 0, \
        logger.info("Data contains education level values outside boundaries.")


def test_valid_relationship_types(data):
    existing_relationship_values = [
        'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']

    relationships = set(data['relationship'].unique())

    logger.info(f'known relations: {relationships}')
    assert set(existing_relationship_values) == set(relationships), \
        logger.info("Invalid relationships found on file.")