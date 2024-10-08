import os
import glob
import pandas as pd

Model_Scaler_Folder = "../models/scaler/"
Model_PCA_Folder = "../models/pca/"

Ticker_Hyperparams_Model_Metrics_Csv = "hyperparameters_search_models/ticker_hyperparams_model_metrics.csv"
Ticker_Trained_Model_Metrics_Csv = "trained_models/ticker_trained_model_metrics.csv"

Hyperparameters_Search_Models_Folder = "hyperparameters_search_models/"
Trained_Models_Folder = "trained_models/"

Hyperparameters_Search_Feature_Importance_Folder = "hyperparameters_search_models/feature_importance/"
Trained_Models_Feature_Importance_Folder = "trained_models/feature_importance/"

Trained_Feature_Folder ='../data/trained_feature/'
PCA_Folder = '../data/pca/'

def make_all_directory():

    os.makedirs('../models/scaler/hyperparameters_search_models/xgbrfregressor', exist_ok=True)
    os.makedirs('../models/scaler/hyperparameters_search_models/xgbregressor_gbtree', exist_ok=True)
    os.makedirs('../models/scaler/hyperparameters_search_models/grnn_regression', exist_ok=True)
    os.makedirs('../models/scaler/hyperparameters_search_models/conv1d_regression', exist_ok=True)
    os.makedirs('../models/scaler/hyperparameters_search_models/lstm_regression', exist_ok=True)
    os.makedirs('../models/scaler/hyperparameters_search_models/transformer_regression', exist_ok=True)
    os.makedirs('../models/scaler/hyperparameters_search_models/conv1d_lstm_regression', exist_ok=True)

    os.makedirs('../models/scaler/trained_models/xgbrfregressor', exist_ok=True)
    os.makedirs('../models/scaler/trained_models/xgbregressor_gbtree', exist_ok=True)
    os.makedirs('../models/scaler/trained_models/grnn_regression', exist_ok=True)
    os.makedirs('../models/scaler/trained_models/conv1d_regression', exist_ok=True)
    os.makedirs('../models/scaler/trained_models/lstm_regression', exist_ok=True)
    os.makedirs('../models/scaler/trained_models/transformer_regression', exist_ok=True)
    os.makedirs('../models/scaler/trained_models/conv1d_lstm_regression', exist_ok=True)

    os.makedirs('../models/scaler/hyperparameters_search_models/feature_importance/xgbrfregressor', exist_ok=True)
    os.makedirs('../models/scaler/hyperparameters_search_models/feature_importance/xgbregressor_gbtree', exist_ok=True)
    os.makedirs('../models/scaler/trained_models/feature_importance/xgbrfregressor', exist_ok=True)
    os.makedirs('../models/scaler/trained_models/feature_importance/xgbregressor_gbtree', exist_ok=True)

    os.makedirs('../models/pca/hyperparameters_search_models/xgbrfregressor', exist_ok=True)
    os.makedirs('../models/pca/hyperparameters_search_models/xgbregressor_gbtree', exist_ok=True)
    os.makedirs('../models/pca/hyperparameters_search_models/grnn_regression', exist_ok=True)
    os.makedirs('../models/pca/hyperparameters_search_models/conv1d_regression', exist_ok=True)
    os.makedirs('../models/pca/hyperparameters_search_models/lstm_regression', exist_ok=True)
    os.makedirs('../models/pca/hyperparameters_search_models/transformer_regression', exist_ok=True)
    os.makedirs('../models/pca/hyperparameters_search_models/conv1d_lstm_regression', exist_ok=True)

    os.makedirs('../models/pca/trained_models/xgbrfregressor', exist_ok=True)
    os.makedirs('../models/pca/trained_models/xgbregressor_gbtree', exist_ok=True)
    os.makedirs('../models/pca/trained_models/grnn_regression', exist_ok=True)
    os.makedirs('../models/pca/trained_models/conv1d_regression', exist_ok=True)
    os.makedirs('../models/pca/trained_models/lstm_regression', exist_ok=True)
    os.makedirs('../models/pca/trained_models/transformer_regression', exist_ok=True)
    os.makedirs('../models/pca/trained_models/conv1d_lstm_regression', exist_ok=True)

    os.makedirs('../models/pca/hyperparameters_search_models/feature_importance/xgbrfregressor', exist_ok=True)
    os.makedirs('../models/pca/hyperparameters_search_models/feature_importance/xgbregressor_gbtree', exist_ok=True)
    os.makedirs('../models/pca/trained_models/feature_importance/xgbrfregressor', exist_ok=True)
    os.makedirs('../models/pca/trained_models/feature_importance/xgbregressor_gbtree', exist_ok=True)

    os.makedirs('../data/all', exist_ok=True)
    os.makedirs('../data/train', exist_ok=True)
    os.makedirs('../data/test', exist_ok=True)

    os.makedirs('../data/commodities_historical_data/original', exist_ok=True)
    os.makedirs('../data/trained_feature', exist_ok=True)
    os.makedirs('../data/pca', exist_ok=True)
    os.makedirs('../predicted_output/scaler/ticker', exist_ok=True)
    os.makedirs('../predicted_output/pca/ticker', exist_ok=True)



def rename_and_overwrite(old_path, new_path):
    # Check if the new model path already exists
    if os.path.exists(new_path):
        os.remove(new_path)  # Remove the existing file
    os.rename(old_path, new_path)  # Rename the old file to the new path
def load_or_create_ticker_df(csv_file_path):
    # Define the column types
    column_types = {
        "Ticker_Symbol": str,

        "xgbrfregressor_1": float,
        "xgbrfregressor_2": float,
        "xgbrfregressor_3": float,
        "xgbrfregressor_4": float,
        "xgbrfregressor_5": float,

        "xgbregressor_gbtree_1": float,
        "xgbregressor_gbtree_2": float,
        "xgbregressor_gbtree_3": float,
        "xgbregressor_gbtree_4": float,
        "xgbregressor_gbtree_5": float,

        "grnn_regression_1": float,
        "grnn_regression_2": float,
        "grnn_regression_3": float,
        "grnn_regression_4": float,
        "grnn_regression_5": float,

        "conv1d_regression_1": float,
        "conv1d_regression_2": float,
        "conv1d_regression_3": float,
        "conv1d_regression_4": float,
        "conv1d_regression_5": float,

        "lstm_regression_1": float,
        "lstm_regression_2": float,
        "lstm_regression_3": float,
        "lstm_regression_4": float,
        "lstm_regression_5": float,

        "conv1d_lstm_regression_1": float,
        "conv1d_lstm_regression_2": float,
        "conv1d_lstm_regression_3": float,
        "conv1d_lstm_regression_4": float,
        "conv1d_lstm_regression_5": float,

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

        "xgbclassifier_gbtree_1": float,
        "xgbclassifier_gbtree_2": float,
        "xgbclassifier_gbtree_3": float,
        "xgbclassifier_gbtree_4": float,
        "xgbclassifier_gbtree_5": float,

        "grnn_classification_1": float,
        "grnn_classification_2": float,
        "grnn_classification_3": float,
        "grnn_classification_4": float,
        "grnn_classification_5": float,

        "conv1d_classification_1": float,
        "conv1d_classification_2": float,
        "conv1d_classification_3": float,
        "conv1d_classification_4": float,
        "conv1d_classification_5": float,

        "lstm_classification_1": float,
        "lstm_classification_2": float,
        "lstm_classification_3": float,
        "lstm_classification_4": float,
        "lstm_classification_5": float,

        "conv1d_lstm_classification_1": float,
        "conv1d_lstm_classification_2": float,
        "conv1d_lstm_classification_3": float,
        "conv1d_lstm_classification_4": float,
        "conv1d_lstm_classification_5": float,

        "transformer_classification_1": float,
        "transformer_classification_2": float,
        "transformer_classification_3": float,
        "transformer_classification_4": float,
        "transformer_classification_5": float,

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


def delete_hyperparameter_search_model(ticker_symbol, model_type, PCA):

    Root_Folder = Model_Scaler_Folder

    if PCA:
        Root_Folder = Model_PCA_Folder

    model_path_folder = Root_Folder + Hyperparameters_Search_Models_Folder + model_type + '/'

    # Delete model files
    for i in range(1, 6):
        pattern = f"{model_path_folder}{ticker_symbol}_{i}.*"
        for filename in glob.glob(pattern):
            try:
                os.remove(filename)
                print(f"Deleted: {filename}")
            except OSError as e:
                print(f"Error: {filename} : {e.strerror}")

    csv_path = Root_Folder + Ticker_Hyperparams_Model_Metrics_Csv
    if os.path.isfile(csv_path):
        try:
            ticker_df = pd.read_csv(csv_path)
            if ticker_symbol in ticker_df['Ticker_Symbol'].values:
                ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol,
                [f'{model_type}_1', f'{model_type}_2', f'{model_type}_3', f'{model_type}_4', f'{model_type}_5']] \
                    = [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA]
                ticker_df.to_csv(csv_path, index=False)
                print(f"Deleted {ticker_symbol} from {csv_path}")
            else:
                print(f"{ticker_symbol} not found in {csv_path}")
        except Exception as e:
            print(f"Error processing CSV file: {e}")
    else:
        print(f"{csv_path} does not exist")

def load_or_create_ticker_metric_df(csv_file_path):
    # Define the column types
    column_types = {
        "Ticker_Symbol": str,

        "RF_1_Sign_Accuracy": float,
        "RF_2_Sign_Accuracy": float,
        "RF_3_Sign_Accuracy": float,
        "RF_4_Sign_Accuracy": float,
        "RF_5_Sign_Accuracy": float,

        "GBT_1_Sign_Accuracy": float,
        "GBT_2_Sign_Accuracy": float,
        "GBT_3_Sign_Accuracy": float,
        "GBT_4_Sign_Accuracy": float,
        "GBT_5_Sign_Accuracy": float,

        "GRNN_1_Sign_Accuracy": float,
        "GRNN_2_Sign_Accuracy": float,
        "GRNN_3_Sign_Accuracy": float,
        "GRNN_4_Sign_Accuracy": float,
        "GRNN_5_Sign_Accuracy": float,

        "CNN_1_Sign_Accuracy": float,
        "CNN_2_Sign_Accuracy": float,
        "CNN_3_Sign_Accuracy": float,
        "CNN_4_Sign_Accuracy": float,
        "CNN_5_Sign_Accuracy": float,

        "LSTM_1_Sign_Accuracy": float,
        "LSTM_2_Sign_Accuracy": float,
        "LSTM_3_Sign_Accuracy": float,
        "LSTM_4_Sign_Accuracy": float,
        "LSTM_5_Sign_Accuracy": float,

        "CNN_LSTM_1_Sign_Accuracy": float,
        "CNN_LSTM_2_Sign_Accuracy": float,
        "CNN_LSTM_3_Sign_Accuracy": float,
        "CNN_LSTM_4_Sign_Accuracy": float,
        "CNN_LSTM_5_Sign_Accuracy": float,

        "Transformer_1_Sign_Accuracy": float,
        "Transformer_2_Sign_Accuracy": float,
        "Transformer_3_Sign_Accuracy": float,
        "Transformer_4_Sign_Accuracy": float,
        "Transformer_5_Sign_Accuracy": float,

        "RF_1_Profit_Loss": float,
        "RF_2_Profit_Loss": float,
        "RF_3_Profit_Loss": float,
        "RF_4_Profit_Loss": float,
        "RF_5_Profit_Loss": float,

        "GBT_1_Profit_Loss": float,
        "GBT_2_Profit_Loss": float,
        "GBT_3_Profit_Loss": float,
        "GBT_4_Profit_Loss": float,
        "GBT_5_Profit_Loss": float,

        "GRNN_1_Profit_Loss": float,
        "GRNN_2_Profit_Loss": float,
        "GRNN_3_Profit_Loss": float,
        "GRNN_4_Profit_Loss": float,
        "GRNN_5_Profit_Loss": float,

        "CNN_1_Profit_Loss": float,
        "CNN_2_Profit_Loss": float,
        "CNN_3_Profit_Loss": float,
        "CNN_4_Profit_Loss": float,
        "CNN_5_Profit_Loss": float,

        "LSTM_1_Profit_Loss": float,
        "LSTM_2_Profit_Loss": float,
        "LSTM_3_Profit_Loss": float,
        "LSTM_4_Profit_Loss": float,
        "LSTM_5_Profit_Loss": float,

        "CNN_LSTM_1_Profit_Loss": float,
        "CNN_LSTM_2_Profit_Loss": float,
        "CNN_LSTM_3_Profit_Loss": float,
        "CNN_LSTM_4_Profit_Loss": float,
        "CNN_LSTM_5_Profit_Loss": float,

        "Transformer_1_Profit_Loss": float,
        "Transformer_2_Profit_Loss": float,
        "Transformer_3_Profit_Loss": float,
        "Transformer_4_Profit_Loss": float,
        "Transformer_5_Profit_Loss": float,

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
