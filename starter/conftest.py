import joblib
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def data():
    """Function to load cleaned data dataframe."""
    df = pd.read_csv("data/clean_census_data.csv")
    return df


@pytest.fixture(scope="session")
def model():
    """Load trained classifier."""
    clf = joblib.load("model/rfc_model.pkl")
    return clf


@pytest.fixture(scope="session")
def encoder():
    """Load fitted encoder."""
    encoder = joblib.load("model/encoder.pkl")
    return encoder


@pytest.fixture(scope="session")
def test_data():
    """Function to load splitted data."""
    test_data = pd.read_csv("data/sample_test.csv")
    return test_data


@pytest.fixture(scope="session")
def sample_input():
    """Function to load sample input data."""
    sample_data = pd.read_csv("data/sample_input.csv")
    return sample_data


@pytest.fixture(scope="session")
def categorical_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]
