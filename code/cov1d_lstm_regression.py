import torch
import optuna
import json
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import numpy as np
import shutil

from directory_manager import *
from optuna_config import *
from cov1d_lstm import *


def conv1d_lstm_regression_hyperparameters_search(X, y, gpu_available, ticker_symbol):
    device = torch.device('cuda' if gpu_available and torch.cuda.is_available() else 'cpu')

    X = X.to_numpy()
    y = y.to_numpy().reshape(-1, 1)
    X = X.reshape((X.shape[0], 1, -1))

    # Split data into training and validation sets
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def conv1d_lstm_regression_objective(trial):

        in_channels = X_train.shape[1]
        out_channels = trial.suggest_int('out_channels', 16, 128)
        kernel_size = trial.suggest_int('kernel_size', 3, 7)
        num_blocks = trial.suggest_int('num_blocks', 1, 5)
        lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 32, 128)
        l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-2)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-5, 1e-1)
        epochs = 1000
        patience = 10

        model = Conv1DLSTMModel(in_channels, out_channels, kernel_size, num_blocks, lstm_hidden_size, l2_lambda, dropout_rate,
                            classification=False).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
        criterion = nn.MSELoss()

        best_val_rmse = np.inf
        epochs_no_improve = 0

        input_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        target_train = torch.tensor(y_train, dtype=torch.float32).to(device)

        input_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        target_val = torch.tensor(y_val, dtype=torch.float32).to(device)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(input_train)
            loss = criterion(output, target_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_output = model(input_val)
                val_rmse = root_mean_squared_error(target_val.cpu(), val_output.cpu())

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break

        return best_val_rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(conv1d_lstm_regression_objective,  n_trials=10)