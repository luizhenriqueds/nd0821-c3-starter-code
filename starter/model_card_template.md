# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Luiz Henrique Q. Lima created the model on Apr 21, 2022. The model is a Random Forest
Classifier using the default hyperparameters from scikit-learn, except for
`n_estimators` that was set to 150 and `max_depth`, set to 10.

## Intended Use

This model should be used to predict whether a person's income would be greater than
$50k/yr based on census data. The available data contains information about education,
occupation, sex, marital status, etc.

## Training Data

The training data was obtained from the [UCI](https://archive.ics.uci.edu/ml/datasets/census+income)
repository and consists of adult census information. It trains a classification model
to predict whether a person would receive an income >= 50k/year based on his social
attributes.
The target variable is encoded as >50K, <=50K and further transformed into [0, 1].

The list of base attributes is shown below

- age
- workclass
- education
- education-num
- marital-status
- occupation
- relationship
- race
- sex
- capital-gain
- capital-loss
- hours-per-week
- native-country

A full description of the possible values for each attribute and its detailed description
can be found at the repository's website.

## Evaluation Data

The original data set has 32562 rows, and an `80%`-`20%` split was used to break this into a
train and test set. No stratification was done. To use the data for training a `One Hot Encoder`
was used on the features and a `label binarizer` was used on the labels.

## Metrics

The model was evaluated using `precision`, `recall` and `fbeta`. All the metrics were used to evaluate
the model both on general and on slices of the data.

The general metric are listed below

- **Model metrics:**
    - `precision`: 0.7287
    - `recall`: 0.621
    - `fbeta`: 0.670

Evaluating the model on slices of the data helps us catch data/model bias, which is an important
aspect to validate. Below we list some under-represented classes which could potentially cause us
problem when running this model on production.

Work class "never-worked" has only 7 example instances on the dataset and all of them are on the
< 50k class. Therefore, the model learned to predict `0` everytime a new case appears.

```json
{
  "feature": "workclass",
  "value": "Never-worked",
  "metrics": {
    "precision": 1.0,
    "recall": 1.0,
    "fbeta": 1.0
  }
}
```

Another example of under-represented class is "Without-pay", with all the metrics 1.0.

```json
 {
  "feature": "workclass",
  "value": "Without-pay",
  "metrics": {
    "precision": 1.0,
    "recall": 1.0,
    "fbeta": 1.0
  }
}
```

Finally, a potentially more critical issue is when dealing with social/nationality aspect. As shown
below, the metrics for people from China are 50%, which is the point of most confusion for an ML model.

```json
{
  "feature": "native-country",
  "value": "China",
  "metrics": {
    "precision": 0.5,
    "recall": 0.5,
    "fbeta": 0.5
  }
}
```

## Ethical Considerations

In the metrics section we pointed out some potential inherent bias on the data and model training. As
we are dealing with social data on this project, we should be careful in collecting these data and try to
avoid under-represented groups to improve the model robustness.

## Caveats and Recommendations

As recommendation, all predictions should be presented to users with the confidence, and therefore people
using this model should be aware of potential gaps and pitfalls in the predictions. Other than that,
a new version of this project should aim at integrating more diverse social data to improve the model
overall quality as well as quality on minority groups.
