import json

import requests

LIVE_ENDPOINT = "https://udacity-ml-project.herokuapp.com/predict"


def make_post_request():
    sample_payload = {
        "age": 49,
        "workclass": "Self-emp-inc",
        "fnlgt": 191681,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }

    headers = {
        'content-type': "application/json",
        'cache-control': "no-cache"
    }

    response = requests.request(
        "POST",
        LIVE_ENDPOINT,
        data=json.dumps(sample_payload),
        headers=headers
    )
    prediction = json.loads(response.text)["prediction"]

    return prediction, response.status_code


if __name__ == '__main__':
    pred, status_code = make_post_request()
    print(f"Predicted: {pred} with status code: {status_code}")
