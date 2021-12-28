import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_columns_names(data):
    expected_columns = [
        "age",
        "workclass",
        "fnlwgt",
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
        "income"
    ]

    curr_columns = data.columns.values
    assert list(expected_columns) == list(curr_columns), \
        logger.info("Wrong column names.")


def test_age_range(data, min_age=0, max_age=100):
    idx = data['age'].between(min_age, max_age)

    assert np.sum(~idx) == 0, \
        logger.info("Data contains values outside boundaries.")


def test_relationship_category(data):
    known_relationship_values = [
        'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']

    relationships = set(data['relationship'].unique())
    logger.info(f'known relations: {relationships}')
    # Unordered check
    assert set(known_relationship_values) == set(relationships), \
        logger.info("Invalid relationships found on file.")