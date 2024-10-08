import optuna
import json

import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
from torch.utils.data import DataLoader, TensorDataset

from directory_manager import *
from optuna_config import *
from grnn import *
from metric import *

Model_Type = "grnn_regression"


def grnn_regression_hyperparameters_search(X, y, gpu_available, ticker_symbol, pca):
    Root_Folder = Model_Scaler_Folder
    if pca:
        Root_Folder = Model_PCA_Folder

    device = torch.device('cuda' if gpu_available and torch.cuda.is_available() else 'cpu')

    # Convert to tensors directly
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)

    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def grnn_regression_objective(trial):

        # Hyperparameters to tune
        input_size = X_train.shape[1]
        sigma = trial.suggest_float('sigma', 0.01, 1.0)
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-2)
        batch_size = trial.suggest_int('batch_size', 16, 128, log=True)
        epochs = 1000
        patience = 10

        model = GRNN(input_size=input_size, output_size=1, sigma=sigma, classification=False).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
        criterion = nn.MSELoss()

        epochs_no_improve = 0

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        best_val_acc = -np.inf
        for epoch in range(epochs):
            model.train()
            for input_train, target_train in train_loader:
                input_train, target_train = input_train.to(device), target_train.to(device)
                optimizer.zero_grad()
                output = model(input_train)
                loss = criterion(output, target_train)
                loss.backward()
                optimizer.step()

            model.eval()
            val_acc = 0
            with torch.no_grad():
                for input_val, target_val in val_loader:
                    input_val, target_val = input_val.to(device), target_val.to(device)
                    val_output = model(input_val)
                    val_acc += accuracy_torch(target_val.cpu(), val_output.cpu()).item()

            val_acc /= len(val_loader)

            # Report intermediate objective value
            trial.report(val_acc, epoch)

            # Prune unpromising trials
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        return best_val_acc

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(grnn_regression_objective, n_trials=MAX_TRIALS)

    # Get all trials
    all_trials = study.trials

    sorted_trials = sorted(all_trials, key=lambda trial: trial.value, reverse=True)

    metrics = {}
    for i in range(0, 5):
        metrics[f'new_{i}'] = sorted_trials[i].value

    ticker_df = load_or_create_ticker_df(Root_Folder + Ticker_Hyperparams_Model_Metrics_Csv)
    # Check if the ticker_symbol exists
    if ticker_symbol not in ticker_df['Ticker_Symbol'].values:
        # Create a new DataFrame for the new row
        new_row = pd.DataFrame({'Ticker_Symbol': [ticker_symbol]})
        # Concatenate the new row to the existing DataFrame
        ticker_df = pd.concat([ticker_df, new_row], ignore_index=True)

    if ticker_symbol in ticker_df['Ticker_Symbol'].values:
        for i in range(1, 6):
            hyperparameter_model_path = f'{Root_Folder}{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.pth'
            hyperparameter_params_path = f'{Root_Folder}{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.json'
            model_path = f'{Root_Folder}{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.pth'
            params_path = f'{Root_Folder}{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.json'
            column_name = f"{Model_Type}_{i}"
            current_score = ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name].values[0]
            if not pd.isnull(current_score) and \
                    os.path.exists(hyperparameter_model_path) and \
                    os.path.exists(hyperparameter_params_path) and \
                    os.path.exists(model_path) and \
                    os.path.exists(params_path):
                metrics[f'old_{i}'] = current_score

    sorted_metrics = dict(sorted(metrics.items(), key=lambda item: item[1], reverse=True))
    sorted_metrics_list = list(sorted_metrics.items())

    for i in range(4, -1, -1):
        key, value = sorted_metrics_list[i]
        rank = i + 1
        new_model_path = f'{Root_Folder}{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{rank}.pth'
        new_params_path = f'{Root_Folder}{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{rank}.json'

        if key.startswith('old'):
            # Extract the index from the key using split method
            old_index = key.split('_')[1]
            old_model_path = f'{Root_Folder}{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{old_index}.pth'
            old_params_path = f'{Root_Folder}{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{old_index}.json'

            rename_and_overwrite(old_model_path, new_model_path)
            rename_and_overwrite(old_params_path, new_params_path)

        else:
            trial_index = int(key.split('_')[1])
            trial = all_trials[trial_index]
            with open(new_params_path, 'w') as f:
                json.dump(trial.params, f)
            trial_params = trial.params

            input_size = X_train.shape[1]
            epochs = 1000
            patience = 10

            model = GRNN(input_size, 1, trial_params['sigma'], classification=False).to(device)
            optimizer = optim.Adam(model.parameters(), lr=trial_params['lr'], weight_decay=trial_params['l2_lambda'])
            criterion = nn.MSELoss()

            epochs_no_improve = 0

            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=trial_params['batch_size'],
                                      shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=trial_params['batch_size'], shuffle=False)

            best_val_plr = -np.inf
            for epoch in range(epochs):
                model.train()
                for input_train, target_train in train_loader:
                    input_train, target_train = input_train.to(device), target_train.to(device)
                    optimizer.zero_grad()
                    output = model(input_train)
                    loss = criterion(output, target_train)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_plr = 0
                with torch.no_grad():
                    for input_val, target_val in val_loader:
                        input_val, target_val = input_val.to(device), target_val.to(device)
                        val_output = model(input_val)
                        val_plr += accuracy_torch(target_val.cpu(), val_output.cpu()).item()

                val_plr /= len(val_loader)

                if val_plr > best_val_plr:
                    best_val_plr = val_plr
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break
            # Save the new model from trial
            torch.save(model.state_dict(), new_model_path)

        # Update ticker_df with the new metrics
        column_name = f"{Model_Type}_{rank}"
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = value

    # Save the updated ticker_df back to the CSV
    ticker_df.to_csv(Root_Folder + Ticker_Hyperparams_Model_Metrics_Csv, index=False)


def grnn_regression_resume_training(X, y, gpu_available, ticker_symbol, pca=False, hyperparameter_search=False,
                                    delete_old_data=False):
    Root_Folder = Model_Scaler_Folder
    if pca:
        Root_Folder = Model_PCA_Folder

    if delete_old_data:
        delete_hyperparameter_search_model(ticker_symbol, Model_Type, pca)

    hyperparameter_search_needed = False
    for i in range(1, 6):
        hyperparameters_search_model_path = f'{Root_Folder}{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.pth'
        hyperparameters_search_model_params_path = f'{Root_Folder}{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.json'

        if not os.path.exists(hyperparameters_search_model_path) or not os.path.exists(
                hyperparameters_search_model_params_path):
            hyperparameter_search_needed = True
            break

    if hyperparameter_search:
        ticker_df = load_or_create_ticker_df(Root_Folder + Ticker_Hyperparams_Model_Metrics_Csv)
        column_name = f"{Model_Type}_5"
        current_score = ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name].values[0]
        if pd.isnull(current_score) or current_score < ACCEPTABLE_SCORE:
            hyperparameter_search_needed = True

    if hyperparameter_search_needed:
        grnn_regression_hyperparameters_search(X, y, gpu_available, ticker_symbol, pca)

    for i in range(1, 6):
        hyperparameters_search_model_path = f'{Root_Folder}{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.pth'
        trained_model_path = f'{Root_Folder}{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.pth'

        hyperparameters_search_model_params_path = f'{Root_Folder}{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.json'
        trained_model_params_path = f'{Root_Folder}{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.json'

        shutil.copy2(hyperparameters_search_model_path, trained_model_path)
        shutil.copy2(hyperparameters_search_model_params_path, trained_model_params_path)

    hyperparameters_search_model_df = load_or_create_ticker_df(Root_Folder + Ticker_Hyperparams_Model_Metrics_Csv)

    trained_model_df = load_or_create_ticker_df(Root_Folder + Ticker_Trained_Model_Metrics_Csv)
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

    trained_model_df.to_csv(Root_Folder + Ticker_Trained_Model_Metrics_Csv, index=False)

def grnn_regression_predict(X, gpu_available, ticker_symbol, pca=False, no=1):
    device = torch.device('cuda' if gpu_available and torch.cuda.is_available() else 'cpu')

    Root_Folder = Model_Scaler_Folder
    if pca:
        Root_Folder = Model_PCA_Folder

    trained_model_path = f'{Root_Folder}{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{no}.pth'
    trained_model_params_path = f'{Root_Folder}{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{no}.json'



    # Check if the model exists
    if not os.path.exists(trained_model_path) or not os.path.exists(trained_model_params_path):
        return None

    # Load the model parameters from the JSON file
    with open(trained_model_params_path, 'r') as f:
        model_params = json.load(f)

    # Convert directly to tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    input_size = X.shape[1]

    model = GRNN(input_size, 1, model_params['sigma'], classification=False).to(device)

    # Load the model state dict
    model.load_state_dict(torch.load(trained_model_path, map_location=device, weights_only=True), strict=False)
    model.eval()  # Set the model to evaluation mode

    input_tensor = X.to(device)

    # Create a DataLoader for batch processing
    data_loader = DataLoader(TensorDataset(input_tensor), batch_size=model_params['batch_size'], shuffle=False)

    preds_list = []
    with torch.no_grad():  # Disable gradient calculation
        for batch in data_loader:
            batch = batch[0].to(device)  # Get the input tensor from the batch
            preds = model(batch)
            preds_list.append(preds.cpu().numpy())

    # Concatenate all predictions
    preds_numpy = np.concatenate(preds_list, axis=0)
    preds_df = pd.DataFrame(preds_numpy, columns=['Prediction'])

    return preds_df
