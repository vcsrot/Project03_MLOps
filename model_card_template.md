# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf
Model card for classification model used to predict individual's income range.

## Model Details

Random Forest Classifier (Scikit-learn implementation) trained on census data ("census_clean.csv")
Selected parameters: random_state=8, max_depth=16, n_estimators=128

## Intended Use

Object is used to predict individual's income level given required census information. Multiple commercial uses for the application, e.g., risk assessment, segmentation.

## Training Data

Census data from UCI Library (UCI Census Data: https://archive.ics.uci.edu/ml/datasets/census+income)

## Evaluation Data

Model was assessed on a separated split of data, test set with a 0.25 split.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

Metrics considered: Precision, Recall, F-Beta

Overall performance:
Precision: 0.7911301859799714
Recall: 0.5739491437467567
F-Beta score: 0.6652631578947369

## Ethical Considerations

Certain features used could be deemed discriminatory (race, education level) and therefore should be assessed according to the application.

## Caveats and Recommendations

Model could be further refined with feature engineering and hyperparameter tuning.
