import os
import glob
import pandas as pd

Ticker_Hyperparams_Model_Metrics_Csv = "../models/hyperparameters_search_models/ticker_hyperparams_model_metrics.csv"
Ticker_Trained_Model_Metrics_Csv = "../models/trained_models/ticker_trained_model_metrics.csv"

Hyperparameters_Search_Models_Folder = "../models/hyperparameters_search_models/"
Trained_Models_Folder = "../models/trained_models/"
Feature_Importance_Folder = "../feature_importance/"
def make_all_directory():
    os.makedirs('../models/hyperparameters_search_models/conv1d_classification/', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/conv1d_regression/', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/lstm_classification/', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/lstm_regression/', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/transformer_classification/', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/transformer_regression/', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/randomforest_classifier', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/randomforest_regressor', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/xbclassifier', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/xbregressor', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/conv1d_lstm_classification/', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/conv1d_lstm_regression/', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/conv1d_transformer_classification/', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/conv1d_transformer_regression/', exist_ok=True)

    os.makedirs('../models/trained_models/conv1d_classification/', exist_ok=True)
    os.makedirs('../models/trained_models/conv1d_regression/', exist_ok=True)
    os.makedirs('../models/trained_models/lstm_classification/', exist_ok=True)
    os.makedirs('../models/trained_models/lstm_regression/', exist_ok=True)
    os.makedirs('../models/trained_models/transformer_classification/', exist_ok=True)
    os.makedirs('../models/trained_models/transformer_regression/', exist_ok=True)
    os.makedirs('../models/trained_models/randomforest_classifier', exist_ok=True)
    os.makedirs('../models/trained_models/randomforest_regressor', exist_ok=True)
    os.makedirs('../models/trained_models/xbclassifier', exist_ok=True)
    os.makedirs('../models/trained_models/xbregressor', exist_ok=True)
    os.makedirs('../models/trained_models/conv1d_lstm_classification/', exist_ok=True)
    os.makedirs('../models/trained_models/conv1d_lstm_regression/', exist_ok=True)
    os.makedirs('../models/trained_models/conv1d_transformer_classification/', exist_ok=True)
    os.makedirs('../models/trained_models/conv1d_transformer_regression/', exist_ok=True)

    os.makedirs('../feature_importance/randomforest_classifier', exist_ok=True)
    os.makedirs('../feature_importance/randomforest_regressor', exist_ok=True)
    os.makedirs('../feature_importance/xbclassifier', exist_ok=True)
    os.makedirs('../feature_importance/xbregressor', exist_ok=True)

    os.makedirs('../data/train', exist_ok=True)
    os.makedirs('../data/test', exist_ok=True)


def load_or_create_ticker_df(csv_file_path):
    """
    Load the existing ticker DataFrame from a CSV file if it exists,
    otherwise create a new DataFrame with predefined column types.
    Ensure the DataFrame has the specified columns, add any missing columns,
    and rearrange the columns in alphabetical order, excluding 'Ticker_Symbol'.

    Args:
    csv_file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded or newly created DataFrame.
    """
    # Define the column types
    column_types = {
        "Ticker_Symbol": str,

        "conv1d_classification_1": float,
        "conv1d_classification_2": float,
        "conv1d_classification_3": float,
        "conv1d_classification_4": float,
        "conv1d_classification_5": float,

        "conv1d_regression_1": float,
        "conv1d_regression_2": float,
        "conv1d_regression_3": float,
        "conv1d_regression_4": float,
        "conv1d_regression_5": float,

        "lstm_classification_1": float,
        "lstm_classification_2": float,
        "lstm_classification_3": float,
        "lstm_classification_4": float,
        "lstm_classification_5": float,

        "lstm_regression_1": float,
        "lstm_regression_2": float,
        "lstm_regression_3": float,
        "lstm_regression_4": float,
        "lstm_regression_5": float,

        "transformer_classification_1": float,
        "transformer_classification_2": float,
        "transformer_classification_3": float,
        "transformer_classification_4": float,
        "transformer_classification_5": float,

        "transformer_regression_1": float,
        "transformer_regression_2": float,
        "transformer_regression_3": float,
        "transformer_regression_4": float,
        "transformer_regression_5": float,

        "randomforest_classifier_1": float,
        "randomforest_classifier_2": float,
        "randomforest_classifier_3": float,
        "randomforest_classifier_4": float,
        "randomforest_classifier_5": float,

        "randomforest_regressor_1": float,
        "randomforest_regressor_2": float,
        "randomforest_regressor_3": float,
        "randomforest_regressor_4": float,
        "randomforest_regressor_5": float,

        "xbclassifier_1": float,
        "xbclassifier_2": float,
        "xbclassifier_3": float,
        "xbclassifier_4": float,
        "xbclassifier_5": float,

        "xbregressor_1": float,
        "xbregressor_2": float,
        "xbregressor_3": float,
        "xbregressor_4": float,
        "xbregressor_5": float,

        "conv1d_lstm_classification_1": float,
        "conv1d_lstm_classification_2": float,
        "conv1d_lstm_classification_3": float,
        "conv1d_lstm_classification_4": float,
        "conv1d_lstm_classification_5": float,

        "conv1d_lstm_regression_1": float,
        "conv1d_lstm_regression_2": float,
        "conv1d_lstm_regression_3": float,
        "conv1d_lstm_regression_4": float,
        "conv1d_lstm_regression_5": float,

        "conv1d_transformer_classification_1": float,
        "conv1d_transformer_classification_2": float,
        "conv1d_transformer_classification_3": float,
        "conv1d_transformer_classification_4": float,
        "conv1d_transformer_classification_5": float,

        "conv1d_transformer_regression_1": float,
        "conv1d_transformer_regression_2": float,
        "conv1d_transformer_regression_3": float,
        "conv1d_transformer_regression_4": float,
        "conv1d_transformer_regression_5": float,


    }

    if os.path.isfile(csv_file_path):
        # Load the existing file into a DataFrame
        ticker_df = pd.read_csv(csv_file_path)

        # Ensure all specified columns are present
        for column, dtype in column_types.items():
            if column not in ticker_df.columns:
                ticker_df[column] = pd.Series(dtype=dtype)

    else:
        # Create a new DataFrame with the specified column types
        ticker_df = pd.DataFrame(columns=column_types.keys()).astype(column_types)

    return ticker_df


def delete_hyperparameter_search_model(ticker_symbol, model_type):

    model_path_folder = Hyperparameters_Search_Models_Folder + model_type + '/'

    # Delete model files
    for i in range(1, 6):
        pattern = f"{model_path_folder}{ticker_symbol}_{i}.*"
        for filename in glob.glob(pattern):
            try:
                os.remove(filename)
                print(f"Deleted: {filename}")
            except OSError as e:
                print(f"Error: {filename} : {e.strerror}")

    # Update CSV file
    if os.path.isfile(Ticker_Hyperparams_Model_Metrics_Csv):
        try:
            ticker_df = pd.read_csv(Ticker_Hyperparams_Model_Metrics_Csv)
            if ticker_symbol in ticker_df['Ticker_Symbol'].values:
                ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol,
                [f'{model_type}_1', f'{model_type}_2', f'{model_type}_3', f'{model_type}_4', f'{model_type}_5']] \
                    = [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA]
                ticker_df.to_csv(Ticker_Hyperparams_Model_Metrics_Csv, index=False)
                print(f"Deleted {ticker_symbol} from {Ticker_Hyperparams_Model_Metrics_Csv}")
            else:
                print(f"{ticker_symbol} not found in {Ticker_Hyperparams_Model_Metrics_Csv}")
        except Exception as e:
            print(f"Error processing CSV file: {e}")
    else:
        print(f"{Ticker_Hyperparams_Model_Metrics_Csv} does not exist")
