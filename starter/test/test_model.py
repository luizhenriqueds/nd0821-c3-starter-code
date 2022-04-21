import numpy as np


def test_data_shape(data):
    """Asserting clean data has no null values"""
    assert data.shape == data.dropna().shape


def test_data_size(data):
    """Asserting data size is greater than threshold"""
    assert data.shape[0] > 30000


def test_data_types(data, categorical_features):
    """Asserting all categorical columns are present"""
    assert set(data.columns).issuperset(categorical_features)


def test_data_values(data):
    """Assert a certain column contains all possible values"""
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
    """Assert model can output predictions and classes are correct"""
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


def test_input(test_data, model, encoder, categorical_features):
    """Assert prediction from given input is from right class"""
    # Test only first instance, known to be from class 0
    test = test_data.drop(['salary'], axis=1)[:1]

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
