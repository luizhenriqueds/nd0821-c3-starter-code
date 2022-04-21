# Script to train machine learning model.

import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from ml import model
# Add the necessary imports for the starter code.
from ml.data import process_data

# Add code to load in the data.
data = pd.read_csv("../data/clean_census_data.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder_test, lb_test = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

joblib.dump(encoder, "../model/encoder.pkl")

# Training model
clf = model.train_model(X_train, y_train)
joblib.dump(clf, "../model/rfc_model.pkl")

# Computing predictions
preds = model.inference(clf, X_test)

precision, recall, fbeta = model.compute_model_metrics(
    y_test, preds
)

print(
    f"model metrics: \n"
    f"\tprecision: {precision} \n"
    f"\trecall: {recall} \n"
    f"\tfbeta: {fbeta}"
)

reporter = model.compute_slice_metrics(
    test, clf, encoder, lb, cat_features
)

with open("../data/slice_output.txt", 'w') as file:
    file.write(json.dumps(reporter, indent=4))
