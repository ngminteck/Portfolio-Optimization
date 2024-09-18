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

    os.makedirs('../models/hyperparameters_search_models/xgbrfclassifier', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/xgbrfregressor', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/xgbclassifier_gbtree', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/xgbregressor_gbtree', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/xgbclassifier_dart', exist_ok=True)
    os.makedirs('../models/hyperparameters_search_models/xgbregressor_dart', exist_ok=True)

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

    os.makedirs('../models/trained_models/xgbrfclassifier', exist_ok=True)
    os.makedirs('../models/trained_models/xgbrfregressor', exist_ok=True)
    os.makedirs('../models/trained_models/xgbclassifier_gbtree', exist_ok=True)
    os.makedirs('../models/trained_models/xgbregressor_gbtree', exist_ok=True)
    os.makedirs('../models/trained_models/xgbclassifier_dart', exist_ok=True)
    os.makedirs('../models/trained_models/xgbregressor_dart', exist_ok=True)

    os.makedirs('../models/trained_models/conv1d_lstm_classification/', exist_ok=True)
    os.makedirs('../models/trained_models/conv1d_lstm_regression/', exist_ok=True)
    os.makedirs('../models/trained_models/conv1d_transformer_classification/', exist_ok=True)
    os.makedirs('../models/trained_models/conv1d_transformer_regression/', exist_ok=True)

    os.makedirs('../feature_importance/xgbrfclassifier', exist_ok=True)
    os.makedirs('../feature_importance/xgbrfregressor', exist_ok=True)
    os.makedirs('../feature_importance/xgbclassifier_gbtree', exist_ok=True)
    os.makedirs('../feature_importance/xgbregressor_gbtree', exist_ok=True)
    os.makedirs('../feature_importance/xgbclassifier_dart', exist_ok=True)
    os.makedirs('../feature_importance/xgbregressor_dart', exist_ok=True)

    os.makedirs('../data/all', exist_ok=True)
    os.makedirs('../data/train', exist_ok=True)
    os.makedirs('../data/test', exist_ok=True)

    os.makedirs('../data/commodities_historical_data/original', exist_ok=True)

    os.makedirs('../result/ticker', exist_ok=True)

def rename_and_overwrite(old_path, new_path):
    # Check if the new model path already exists
    if os.path.exists(new_path):
        os.remove(new_path)  # Remove the existing file
    os.rename(old_path, new_path)  # Rename the old file to the new path
def load_or_create_ticker_df(csv_file_path):
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

        "xgbrfclassifier_1": float,
        "xgbrfclassifier_2": float,
        "xgbrfclassifier_3": float,
        "xgbrfclassifier_4": float,
        "xgbrfclassifier_5": float,

        "xgbrfregressor_1": float,
        "xgbrfregressor_2": float,
        "xgbrfregressor_3": float,
        "xgbrfregressor_4": float,
        "xgbrfregressor_5": float,

        "xgbclassifier_gbtree_1": float,
        "xgbclassifier_gbtree_2": float,
        "xgbclassifier_gbtree_3": float,
        "xgbclassifier_gbtree_4": float,
        "xgbclassifier_gbtree_5": float,

        "xgbregressor_gbtree_1": float,
        "xgbregressor_gbtree_2": float,
        "xgbregressor_gbtree_3": float,
        "xgbregressor_gbtree_4": float,
        "xgbregressor_gbtree_5": float,

        "xgbclassifier_dart_1": float,
        "xgbclassifier_dart_2": float,
        "xgbclassifier_dart_3": float,
        "xgbclassifier_dart_4": float,
        "xgbclassifier_dart_5": float,

        "xgbregressor_dart_1": float,
        "xgbregressor_dart_2": float,
        "xgbregressor_dart_3": float,
        "xgbregressor_dart_4": float,
        "xgbregressor_dart5": float,

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

def load_or_create_ticker_metric_df(csv_file_path):
    # Define the column types
    column_types = {
        "Ticker_Symbol": str,

        "Top_1_CNN1D_Sign_Accuracy_PERCENT": float,
        "Top_1_CNN1D_Value_RMSE_PERCENT": float,
        "Top_5_CNN1D_Sign_Accuracy_PERCENT": float,
        "Top_5_CNN1D_Value_RMSE_PERCENT": float,

        "Top_1_LSTM_Sign_Accuracy_PERCENT": float,
        "Top_1_LSTM_Value_RMSE_PERCENT": float,
        "Top_5_LSTM_Sign_Accuracy_PERCENT": float,
        "Top_5_LSTM_Value_RMSE_PERCENT": float,

        "Top_1_CNN1D_LSTM_Sign_Accuracy_PERCENT": float,
        "Top_1_CNN1D_LSTM_Value_RMSE_PERCENT": float,
        "Top_5_CNN1D_LSTM_Sign_Accuracy_PERCENT": float,
        "Top_5_CNN1D_LSTM_Value_RMSE_PERCENT": float,

        "Top_1_RF_Sign_Accuracy_PERCENT": float,
        "Top_1_RF_Value_RMSE_PERCENT": float,
        "Top_5_RF_Sign_Accuracy_PERCENT": float,
        "Top_5_RF_Value_RMSE_PERCENT": float,

        "Top_1_GT_Sign_Accuracy_PERCENT": float,
        "Top_1_GT_Value_RMSE_PERCENT": float,
        "Top_5_GT_Sign_Accuracy_PERCENT": float,
        "Top_5_GT_Value_RMSE_PERCENT": float,

        "Top_1_Combined_Sign_Accuracy_PERCENT": float,
        "Top_1_Combined_Value_RMSE_PERCENT": float,
        "Top_5_Combined_Sign_Accuracy_PERCENT": float,
        "Top_5_Combined_Value_RMSE_PERCENT": float,

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
