import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.starter.ml.data import process_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s"
)
logger = logging.getLogger()


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Train and save a model.
    rfc = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    logger.info(f"{__name__} - Training model")
    rfc.fit(X_train, y_train)

    return rfc


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    logger.info(f"{__name__} - Computing model metrics")
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_slice_metrics(data, model, encoder, lb, categorical_features):
    """
    Validates the trained machine learning model on slices of
    the data using precision, recall, and F1.

    Inputs
    ------
    reporter : pd.Dataframe
        Test data to be used in slicing.
    model : RandomForestClassifier
        Trained classifier to perform predictions.
    encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Fitted encoder.
    lb: sklearn.preprocessing._label.LabelBinarizer
        Fitted binarizer.
    categorical_features: list
        Categorical features to compute slices.
    Returns
    -------
    reporter : dict
        A metric reporter with the slice metrics for all
    categorical features
    """
    logger.info(f"{__name__} - Computing model metrics on slice")

    reporter = []
    for col in categorical_features:
        cat_values = data[col].unique()
        for value in cat_values:
            logger.info(f"\n{__name__} Measure performance on feature {col}")
            filtered_test = data[data[col] == value]
            X_test, y_test, _, _ = process_data(
                filtered_test,
                categorical_features=categorical_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb
            )

            preds = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(
                y_test, preds
            )
            metrics = {
                "feature": col,
                "value": value,
                "metrics": {
                    "precision": round(precision, 5),
                    "recall": round(recall, 5),
                    "fbeta": round(fbeta, 5)
                }
            }
            reporter.append(metrics)
    return reporter


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    logger.info(f"{__name__} - Computing model predictions")
    return model.predict(X)
