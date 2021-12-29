from starter.train_model import get_data, create_model, batch_inference

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

if __name__ == '__main__':
    data_path = 'data/census_clean.csv'
    model_path = "model/random_forest_model.pkl"
    print(model_path)

    # Process data:
    train_data, test_data = get_data(data_path)

    # Training the model on the train data
    create_model(train_data, CAT_FEATURES, model_path)

    # Generate model performace on test set:
    precision, recall, f_beta = batch_inference(test_data,
                                                model_path,
                                                CAT_FEATURES)