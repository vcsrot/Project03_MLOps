import pandas as pd
from starter.train_model import batch_inference
from model_train_pipeline import CAT_FEATURES


def create_data_slice(data_path, col_to_slice, value_to_replace=None):

    # Add code to load in the data.
    if value_to_replace:
        input_df = pd.read_csv(data_path, index_col=None)
        input_df[col_to_slice] = input_df[col_to_slice].apply(
            lambda x: str(value_to_replace)
        )

    else:
        input_df = pd.read_csv(data_path, index_col=None)
        input_df[col_to_slice] = input_df[col_to_slice].apply(
            lambda x: input_df[col_to_slice][0]
        )

    return input_df


if __name__ == '__main__':
    col_fold = 'education'
    # category = 'Bachelors'

    df = pd.read_csv('data/census_clean.csv', index_col=None)

    with open('slice_output.txt', 'a') as f:

        for item in df[col_fold].unique():

            print("Performance on column: {} - category: {}".format(col_fold, item))
            sliced_data = create_data_slice('data/census_clean.csv',
                                            col_fold,
                                            item)

            precision, recall, fbeta = batch_inference(df[df[col_fold] == item],
                                                    "model/random_forest_model.pkl",
                                                    CAT_FEATURES)

            result = f"""\nPerformance on sliced feature -- {col_fold} -- {item} \
            \nPrecision:\t{precision}\nRecall:\t{recall}\nF-beta score:\t{fbeta}\n"""
            
            f.write(result)