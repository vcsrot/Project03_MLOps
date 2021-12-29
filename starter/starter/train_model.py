# Script to train machine learning model.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Additional dependencies
from .ml.data import process_data
from .ml.model import compute_model_metrics, inference, train_model
import joblib


# Add code to load in the data.
def get_data(data_path):
    input_df = pd.read_csv(data_path, index_col=None)

    # Split data
    train_data, test_data = train_test_split(
        input_df, test_size=0.25, random_state=42, shuffle=True
    )

    return train_data, test_data

def create_model(train_data, categorical_features, model_path, label_column='salary'):
    # Proces the train data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train_data, categorical_features, label="salary", training=True
    )

    # Train and save model
    model = train_model(X_train, y_train)

    # Save the model in `model_path`
    joblib.dump((model, encoder, lb), model_path)


def batch_inference(test_data, model_path, cat_features, label_column='salary'):
    # Load trained object from path:
    model, encoder, lb = joblib.load(model_path)

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test_data,
        categorical_features=cat_features,
        label=label_column,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Generate metrics
    preds = inference(model=model, X=X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F-Beta score: {}'.format(fbeta))

    return precision, recall, fbeta


def online_inference(row_dict, model_path, cat_features):
    # Load trained object from path:
    model, encoder, lb = joblib.load(model_path)

    # Assert data types:
    row_transformed = list()
    X_categorical = list()
    X_continuous = list()

    # Iterate over input dictionary and append values:
    for key, value in row_dict.items():
        mod_key = key.replace('_', '-')
        if mod_key in cat_features:
            X_categorical.append(value)
        else:
            X_continuous.append(value)

    # Transform data input:
    y_cat = encoder.transform([X_categorical])
    y_conts = np.asarray([X_continuous])

    row_transformed = np.concatenate([y_conts, y_cat], axis=1)

    # Generate inference with model:
    predictions = inference(model=model, X=row_transformed)

    return '>50K' if predictions[0] else '<=50K'
