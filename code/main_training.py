import xgboost

from data_preprocessing import *

from xgbrfregressor import *
from xgbregressor_gbtree import *
from grnn_regression import *
from cov1d_regression import *
from lstm_regression import *
from cov1d_lstm_regression import *
from transformer_regression import *


def main_training(classification = False):
    logical_cores = os.cpu_count()
    print(f"Number of logical CPU cores: {logical_cores}")

    num_workers = max(1, logical_cores // 2)
    print(f"Number of workers set to: {num_workers}")

    def is_gpu_available():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    gpu_available = is_gpu_available()
    print(f"GPU available: {gpu_available}")

    print(xgboost.build_info())

    path = '../data/all'

    ticker_list = []

    if os.path.exists(path):
        ticker_list = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.csv')]

    for ticker_symbol in ticker_list:

        X, y_regressor = training_preprocess_data(ticker_symbol)


        # If you want to perform hyperparameter search and update the existing model:
        # Example: conv1d_classification_resume_training(X, y_classifier, gpu_available, ticker_symbol, True)

        # If you want to start hyperparameter search from the beginning and delete old records (mostly needed if features and target have been changed):
        # Example: conv1d_classification_resume_training(X, y_classifier, gpu_available, ticker_symbol, True, True)

        xgbregressor_gbtree_resume_training(X, y_regressor, gpu_available, ticker_symbol)
        xgbrfregressor_resume_training(X, y_regressor, gpu_available, ticker_symbol)
        grnn_regression_resume_training(X, y_regressor, gpu_available, ticker_symbol)
        conv1d_regression_resume_training(X, y_regressor, gpu_available, ticker_symbol, True)
        lstm_regression_resume_training(X, y_regressor, gpu_available, ticker_symbol, True)
        conv1d_lstm_regression_resume_training(X, y_regressor, gpu_available, ticker_symbol, True)
        #transformer_regression_resume_training(X, y_regressor, gpu_available, ticker_symbol)




        print(f"{ticker_symbol} done training.")











