import numpy as np


def test_data_shape(data):
    """Asserting clean data has no null values"""
    assert data.shape == data.dropna().shape


def test_data_size(data):
    """Asserting data size is greater than threshold"""
    assert data.shape[0] > 30000


def test_data_types(data):
    """Asserting all categorical columns are present"""

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    assert set(data.columns).issuperset(cat_features)


def test_data_values(data):
    known_education_values = [
        'State-gov',
        'Self-emp-not-inc',
        'Private',
        'Federal-gov',
        'Local-gov',
        '?',
        'Self-emp-inc',
        'Without-pay',
        'Never-worked'
    ]

    assert data['workclass'].isin(known_education_values).all()


def test_data_age(data):
    assert data['age'].dropna().between(17, 90).all()


def test_model_inference(test_data, model, encoder, categorical_features):
    test = test_data.drop(['salary'], axis=1)

    X_categorical = encoder.transform(
        test[categorical_features]
    )

    X_continuous = test.drop(
        *[categorical_features], axis=1
    )

    X = np.concatenate(
        [
            X_continuous,
            X_categorical
        ], axis=1
    )

    preds = model.predict(X[:10])

    assert len(preds) == 10
    assert np.isin(preds, [0, 1]).all()


def test_input(sample_input, model, encoder, categorical_features):
    test = sample_input.drop(['salary'], axis=1)

    X_categorical = encoder.transform(
        test[categorical_features]
    )

    X_continuous = test.drop(
        *[categorical_features], axis=1
    )

    X = np.concatenate(
        [
            X_continuous,
            X_categorical
        ], axis=1
    )

    pred = model.predict(X)
    # Label is 0
    assert pred == 0
