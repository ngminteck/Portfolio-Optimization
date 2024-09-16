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

Model_Type = "conv1d_lstm_regression"

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

    # Get all trials
    all_trials = study.trials

    # Sort trials by their objective values in ascending order
    sorted_trials = sorted(all_trials, key=lambda trial: trial.value)

    metrics = {}
    for i in range(0, 5):
        metrics[f'new_{i}'] = sorted_trials[i].value

    ticker_df = load_or_create_ticker_df(Ticker_Hyperparams_Model_Metrics_Csv)
    # Check if the ticker_symbol exists
    if ticker_symbol not in ticker_df['Ticker_Symbol'].values:
        # Create a new DataFrame for the new row
        new_row = pd.DataFrame({'Ticker_Symbol': [ticker_symbol]})
        # Concatenate the new row to the existing DataFrame
        ticker_df = pd.concat([ticker_df, new_row], ignore_index=True)

    if ticker_symbol in ticker_df['Ticker_Symbol'].values:
        for i in range(1, 6):
            column_name = f"{Model_Type}_{i}"
            current_score = ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name].values[0]
            if not pd.isnull(current_score):
                metrics[f'old_{i}'] = current_score

    sorted_metrics = dict(sorted(metrics.items(), key=lambda item: item[1]))
    sorted_metrics_list = list(sorted_metrics.items())

    for i in range(4, -1, -1):
        key, value = sorted_metrics_list[i]
        rank = i + 1
        new_model_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{rank}.pth'
        new_params_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{rank}.json'

        if key.startswith('old'):
            # Extract the index from the key using split method
            old_index = key.split('_')[1]
            old_model_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{old_index}.pth'
            old_params_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{old_index}.json'

            rename_and_overwrite(old_model_path, new_model_path)
            rename_and_overwrite(old_params_path, new_params_path)

        else:
            trial_index = int(key.split('_')[1])
            trial = all_trials[trial_index]
            trial_params = trial.params

            in_channels = X_train.shape[1]
            epochs = 1000
            patience = 10

            model = Conv1DLSTMModel(in_channels, trial_params['out_channels'], trial_params['kernel_size'],  trial_params['num_blocks'], trial_params['lstm_hidden_size'], trial_params['l2_lambda'], trial_params['dropout_rate'],classification=False).to(device)

            optimizer = optim.Adam(model.parameters(), lr=trial_params['lr'], weight_decay=trial_params['l2_lambda'])
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

            # Save the new model from trial

            with open(new_params_path, 'w') as f:
                json.dump(trial.params, f)
            torch.save(model.state_dict(), new_model_path)

        # Update ticker_df with the new metrics
        column_name = f"{Model_Type}_{rank}"
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = value

    # Save the updated ticker_df back to the CSV
    ticker_df.to_csv(Ticker_Hyperparams_Model_Metrics_Csv, index=False)

def conv1d_lstm_regression_resume_training(X, y, gpu_available, ticker_symbol, hyperparameter_search=False,
                                          delete_old_data=False):

    if delete_old_data:
        delete_hyperparameter_search_model(ticker_symbol, Model_Type)

    all_existed = True
    for i in range(1, 6):
        hyperparameters_search_model_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.pth'
        params_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.json'

        if not os.path.exists(hyperparameters_search_model_path) or not os.path.exists(params_path):
            all_existed = False
            break

    if not all_existed or hyperparameter_search:
        conv1d_lstm_regression_hyperparameters_search(X, y, gpu_available, ticker_symbol)

    for i in range(1, 6):
        hyperparameters_search_model_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.pth'
        trained_model_path = f'{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.pkl'
        shutil.copy2(hyperparameters_search_model_path, trained_model_path)

    hyperparameters_search_model_df = load_or_create_ticker_df(Ticker_Hyperparams_Model_Metrics_Csv)

    trained_model_df = load_or_create_ticker_df(Ticker_Trained_Model_Metrics_Csv)
    if ticker_symbol not in trained_model_df['Ticker_Symbol'].values:
        # Create a new DataFrame for the new row
        new_row = pd.DataFrame({'Ticker_Symbol': [ticker_symbol]})
        # Concatenate the new row to the existing DataFrame
        trained_model_df = pd.concat([trained_model_df, new_row], ignore_index=True)

    # List of columns to copy
    columns_to_copy = [f'{Model_Type}_1', f'{Model_Type}_2', f'{Model_Type}_3', f'{Model_Type}_4',
                       f'{Model_Type}_5']

    # Copy the values from hyperparameters_search_model_df to trained_model_df for the specific row
    for column in columns_to_copy:
        trained_model_df.loc[trained_model_df['Ticker_Symbol'] == ticker_symbol, column] = \
            hyperparameters_search_model_df.loc[
                hyperparameters_search_model_df['Ticker_Symbol'] == ticker_symbol, column].values[0]

    trained_model_df.to_csv(Ticker_Trained_Model_Metrics_Csv, index=False)