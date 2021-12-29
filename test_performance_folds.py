import pandas as pd
from starter.train_model import batch_inference
from main import CAT_FEATURES


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
    category = 'Bachelors'

    print("Performance on column: {} - category: {}".format(col_fold, category))
    sliced_data = create_data_slice('data/census_clean.csv',
                                    col_fold,
                                    category)

    precision, recall, fbeta = batch_inference(sliced_data,
                                               "model/random_forest_model.pkl",
                                               CAT_FEATURES)

    with open('Performance_Fold_{}_{}.txt'.format(col_fold, category), 'a') as f:
        result = f"""\n{'-'*50}\nperformance on sliced column -- {col_fold} -- {category}\n{'-'*50} \
            \nPrecision:\t{precision}\nRecall:\t{recall}\nF-beta score:\t{fbeta}\n"""
        f.write(result)