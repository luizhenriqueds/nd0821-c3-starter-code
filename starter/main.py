# Put the code for your API here.

import http
import json
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi import Response
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    remote = os.getenv("REMOTE")
    url = os.getenv("URL")
    os.system("dvc config core.no_scm true")
    os.system(f"dvc remote add -d {remote} {url}")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


class Payload(BaseModel):
    age: int = Field(alias="age", example=39)
    workclass: str = Field(alias="workclass", example="State-gov")
    fnlgt: int = Field(alias="fnlgt", example=77516)
    education: str = Field(alias="education", example="Bachelors")
    education_num: int = Field(alias="education-num", example=13)
    marital_status: str = Field(alias="marital-status", exaple="Never-married")
    occupation: str = Field(alias="occupation", example="Adm-clerical")
    relationship: str = Field(alias="relationship", example="Not-in-family")
    race: str = Field(alias="race", example="While")
    sex: str = Field(alias="sex", exemple="Male")
    capital_gain: int = Field(alias="capital-gain", example=2174)
    capital_loss: int = Field(alias="capital-loss", example=0)
    hours_per_week: int = Field(alias="hours-per-week", example=40)
    native_country: str = Field(alias="native-country", example="United-States")


def _load_artifacts():
    BASE_PATH = "starter/model"
    clf = joblib.load(f"{BASE_PATH}/rfc_model.pkl")
    encoder = joblib.load(f"{BASE_PATH}/encoder.pkl")

    return clf, encoder


@app.post("/predict", response_model=Payload)
async def inference(payload: Payload):
    clf, encoder = _load_artifacts()

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    input_payload = pd.DataFrame(jsonable_encoder(payload), index=[0])

    X_categorical = encoder.transform(input_payload[categorical_features])
    X_continuous = input_payload.drop(*[categorical_features], axis=1)

    X = np.concatenate(
        [
            X_continuous,
            X_categorical
        ], axis=1
    )

    prediction = clf.predict(X)
    encoded_pred = "<= 50k/yr" if prediction == 0 else "> 50k/yr"

    return Response(
        content=json.dumps({
            "model_version": 1,
            "prediction": encoded_pred
        }),
        status_code=http.HTTPStatus.OK,
        media_type="application/json"
    )
