import os
import pandas as pd


def make_all_directory():
    os.makedirs('../models/hyperparameters-search-models/pytorch/conv1d-classification/', exist_ok=True)
    os.makedirs('../models/hyperparameters-search-models/pytorch/conv1d-regression/', exist_ok=True)
    os.makedirs('../models/hyperparameters-search-models/pytorch/lstm-classification/', exist_ok=True)
    os.makedirs('../models/hyperparameters-search-models/pytorch/lstm-regression/', exist_ok=True)
    os.makedirs('../models/hyperparameters-search-models/pytorch/transformer-classification/', exist_ok=True)
    os.makedirs('../models/hyperparameters-search-models/pytorch/transformer-regression/', exist_ok=True)
    os.makedirs('../models/hyperparameters-search-models/xgboost/xbclassifier', exist_ok=True)
    os.makedirs('../models/hyperparameters-search-models/xgboost/xbregressor', exist_ok=True)

    os.makedirs('../models/best-hyperparameters/pytorch/conv1d-classification/', exist_ok=True)
    os.makedirs('../models/best-hyperparameters/pytorch/conv1d-regression/', exist_ok=True)
    os.makedirs('../models/best-hyperparameters/pytorch/lstm-classification/', exist_ok=True)
    os.makedirs('../models/best-hyperparameters/pytorch/lstm-regression/', exist_ok=True)
    os.makedirs('../models/best-hyperparameters/pytorch/transformer-classification/', exist_ok=True)
    os.makedirs('../models/best-hyperparameters/pytorch/transformer-regression/', exist_ok=True)
    os.makedirs('../models/best-hyperparameters/xgboost/xbclassifier', exist_ok=True)
    os.makedirs('../models/best-hyperparameters/xgboost/xbregressor', exist_ok=True)

    os.makedirs('../models/trained-models/pytorch/conv1d-classification/', exist_ok=True)
    os.makedirs('../models/trained-models/pytorch/conv1d-regression/', exist_ok=True)
    os.makedirs('../models/trained-models/pytorch/lstm-classification/', exist_ok=True)
    os.makedirs('../models/trained-models/pytorch/lstm-regression/', exist_ok=True)
    os.makedirs('../models/trained-models/pytorch/transformer-classification/', exist_ok=True)
    os.makedirs('../models/trained-models/pytorch/transformer-regression/', exist_ok=True)
    os.makedirs('../models/trained-models/xgboost/xbclassifier', exist_ok=True)
    os.makedirs('../models/trained-models/xgboost/xbregressor', exist_ok=True)

    os.makedirs('../feature-importances/xbclassifier', exist_ok=True)
    os.makedirs('../feature-importances/xbregressor', exist_ok=True)

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
        "Best_Cov1D_Classification_Accuracy": float,
        "Best_Cov1D_Classification_Path": str,
        "Best_Cov1D_Regression_RMSE": float,
        "Best_Cov1D_Regression_Path": str,
        "Best_LSTM_Classification_Accuracy": float,
        "Best_LSTM_Classification_Path": str,
        "Best_LSTM_Regression_RMSE": float,
        "Best_LSTM_Regression_Path": str,
        "Best_Transformer_Classification_Accuracy": float,
        "Best_Transformer_Classification_Path": str,
        "Best_Transformer_Regression_RMSE": float,
        "Best_Transformer_Regression_Path": str,
        "Best_XGBClassifier_Classification_Accuracy": float,
        "Best_XGBClassifier_Classification_Path": str,
        "Best_XGBRegressor_Regression_RMSE": float,
        "Best_XGBRegressor_Regression_Path": str
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
    csv_path = '../models/hyperparameters-search-models/ticker-all-models-best-hyperparameters-list.csv'

    conv1d_classification_model_path = f'../models/hyperparameters-search-models/pytorch/{model_type}/{ticker_symbol}.pth'
    conv1d_regression_model_path = f'../models/hyperparameters-search-models/pytorch/{model_type}/{ticker_symbol}.pth'
    lstm_classification_model_path = f'../models/hyperparameters-search-models/pytorch/{model_type}/{ticker_symbol}.pth'
    lstm_regression_model_path = f'../models/hyperparameters-search-models/pytorch/{model_type}/{ticker_symbol}.pth'
    xbclassifier_model_path = f'../models/hyperparameters-search-models/xgboost/{model_type}/{ticker_symbol}.pkl'
    xbregressor_model_path = f'../models/hyperparameters-search-models/xgboost/{model_type}/{ticker_symbol}.pkl'

    conv1d_classification_params_path = f'../models/best-hyperparameters/pytorch/{model_type}/{ticker_symbol}.json'
    conv1d_regression_params_path = f'../models/best-hyperparameters/pytorch/{model_type}/{ticker_symbol}.json'
    lstm_classification_params_path = f'../models/best-hyperparameters/pytorch/{model_type}/{ticker_symbol}.json'
    lstm_regression_params_path = f'../models/best-hyperparameters/pytorch/{model_type}/{ticker_symbol}.json'
    xbclassifier_params_path = f'../models/best-hyperparameters/xgboost/{model_type}/{ticker_symbol}.json'
    xbregressor_params_path = f'../models/best-hyperparameters/xgboost/{model_type}/{ticker_symbol}.json'

    if model_type == "conv1d-classification":
        if os.path.isfile(conv1d_classification_model_path):
            os.remove(conv1d_classification_model_path)
            print(f"Deleted {conv1d_classification_model_path}")
        if os.path.isfile(conv1d_classification_params_path):
            os.remove(conv1d_classification_params_path)
            print(f"Deleted {conv1d_classification_params_path}")

    if model_type == "conv1d-regression":
        if os.path.isfile(conv1d_regression_model_path):
            os.remove(conv1d_regression_model_path)
            print(f"Deleted {conv1d_regression_model_path}")
        if os.path.isfile(conv1d_regression_params_path):
            os.remove(conv1d_regression_params_path)
            print(f"Deleted {conv1d_regression_params_path}")

    if model_type == "lstm-classification":
        if os.path.isfile(lstm_classification_model_path):
            os.remove(lstm_classification_model_path)
            print(f"Deleted {lstm_classification_model_path}")
        if os.path.isfile(lstm_classification_params_path):
            os.remove(lstm_classification_params_path)
            print(f"Deleted {lstm_classification_params_path}")

    if model_type == "lstm-regression":
        if os.path.isfile(lstm_regression_model_path):
            os.remove(lstm_regression_model_path)
            print(f"Deleted {lstm_regression_model_path}")
        if os.path.isfile(lstm_regression_params_path):
            os.remove(lstm_regression_params_path)
            print(f"Deleted {lstm_regression_params_path}")

    if model_type == "xbclassifier":
        if os.path.isfile(xbclassifier_model_path):
            os.remove(xbclassifier_model_path)
            print(f"Deleted {xbclassifier_model_path}")
        if os.path.isfile(xbclassifier_params_path):
            os.remove(xbclassifier_params_path)
            print(f"Deleted {xbclassifier_params_path}")

    if model_type == "xbregressor":
        if os.path.isfile(xbregressor_model_path):
            os.remove(xbregressor_model_path)
            print(f"Deleted {xbregressor_model_path}")
        if os.path.isfile(xbregressor_params_path):
            os.remove(xbregressor_params_path)
            print(f"Deleted {xbregressor_params_path}")

    if os.path.isfile(csv_path):
        ticker_df = pd.read_csv(csv_path)
        if ticker_symbol in ticker_df['Ticker_Symbol'].values:
            if model_type == "conv1d-classification":
                ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, ['Best_Cov1D_Classification_Accuracy',
                                                                            'Best_Cov1D_Classification_Path']] = [pd.NA,
                                                                                                                  pd.NA]
            if model_type == "conv1d-regression":
                ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, ['Best_Cov1D_Regression_RMSE',
                                                                            'Best_Cov1D_Regression_Path']] = [pd.NA,
                                                                                                              pd.NA]
            if model_type == "lstm-classification":
                ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, ['Best_LSTM_Classification_Accuracy',
                                                                            'Best_LSTM_Classification_Path']] = [pd.NA,
                                                                                                                 pd.NA]
            if model_type == "lstm-regression":
                ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, ['Best_LSTM_Regression_RMSE',
                                                                            'Best_LSTM_Regression_Path']] = [pd.NA,
                                                                                                             pd.NA]
            if model_type == "xbclassifier":
                ticker_df.loc[
                    ticker_df['Ticker_Symbol'] == ticker_symbol, ['Best_XGBClassifier_Classification_Accuracy',
                                                                  'Best_XGBClassifier_Classification_Path']] = [pd.NA,
                                                                                                                pd.NA]
            if model_type == "xbregressor":
                ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, ['Best_XGBRegressor_Regression_RMSE',
                                                                            'Best_XGBRegressor_Regression_Path']] = [
                    pd.NA, pd.NA]

            ticker_df.to_csv(csv_path, index=False)
            print(f"Deleted {ticker_symbol} from {csv_path}")
        else:
            print(f"{ticker_symbol} not found in {csv_path}")
    else:
        print(f"{csv_path} does not exist")
