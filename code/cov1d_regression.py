import torch
import optuna
import json
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import numpy as np

from directory_manager import *
from cov1d import *

Model_Type = "cov1d_regression"
def conv1d_regression_objective(X, y, gpu_available, trial):
    X = X.to_numpy()
    y = y.to_numpy().reshape(-1, 1)

    # Reshape X for Conv1D
    NUM_CHANNELS = 1
    X = X.reshape((X.shape[0], NUM_CHANNELS, -1))  # Reshape for Conv1D: (batch_size, num_channels, sequence_length)

    # Split data into training and validation sets
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    device = torch.device('cuda' if gpu_available and torch.cuda.is_available() else 'cpu')

    in_channels = X_train.shape[1]
    out_channels = trial.suggest_int('out_channels', 16, 128)
    kernel_size = trial.suggest_int('kernel_size', 3, 7)
    num_blocks = trial.suggest_int('num_blocks', 1, 5)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-2)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    epochs = 1000

    model = Conv1DModel(in_channels, out_channels, kernel_size, num_blocks, l2_lambda, dropout_rate,
                        classification=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float('lr', 1e-5, 1e-2), weight_decay=l2_lambda)
    criterion = nn.MSELoss()

    patience = 10
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
                print(f"Early stopping at epoch {epoch}")
                break

    return best_val_rmse
def conv1d_regression_hyperparameters_search(X, y, gpu_available, ticker_symbol, delete_old_data=False):
    if delete_old_data:
        delete_hyperparameter_search_model(ticker_symbol, "conv1d-regression")

    study = optuna.create_study(direction='minimize')
    study.optimize(conv1d_regression_objective, X, y, gpu_available, n_trials=100)

    top_trials = study.trials_dataframe().sort_values(by='value', ascending=True).head(5)

    metrics = {}

    for i, trial in top_trials.iterrows():
        metrics[f'new_{i}'] = trial['value']

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

    sorted_metrics = {k: v for k, v in sorted(metrics.items(), key=lambda item: item[1], reverse=False)}

    sorted_items = list(sorted_metrics.items())

    for rank, (key, value) in enumerate(sorted_items[:5], start=1):
        new_model_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{rank}.pth'
        new_params_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{rank}.json'

        if key.startswith('old'):
            # Extract the index from the key using split method
            old_index = key.split('_')[1]
            old_model_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{old_index}.pth'
            old_params_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{old_index}.json'

            os.rename(old_model_path, new_model_path)
            os.rename(old_params_path, new_params_path)

        else:
            # Extract the trial index from the key using split method
            trial_index = int(key.split('_')[1])
            trial = top_trials.iloc[trial_index]

            # Save the new model from trial
            torch.save(trial.user_attrs['model'].state_dict(), new_model_path)
            with open(new_params_path, 'w') as f:
                json.dump(trial.params, f)

        # Update ticker_df with the new metrics
        column_name = f"{Model_Type}_{rank}"
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = value

    # Save the updated ticker_df back to the CSV
    ticker_df.to_csv(Ticker_Hyperparams_Model_Metrics_Csv, index=False)