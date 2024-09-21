import optuna
import json
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import shutil
from torch.utils.data import DataLoader, TensorDataset

from directory_manager import *
from optuna_config import *
from lstm import *
from sequence_length import *

Model_Type = "lstm_regression"

def lstm_regression_hyperparameters_search(X, y, gpu_available, ticker_symbol):
    device = torch.device('cuda' if gpu_available and torch.cuda.is_available() else 'cpu')

    # Convert directly to tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)

    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    def lstm_regression_objective(trial):
        sequence_length = trial.suggest_int('sequence_length', 2, 30)
        hidden_size = trial.suggest_int('hidden_size', 16, 128)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        num_blocks = trial.suggest_int('num_blocks', 1, 3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-5, 1e-1)
        batch_size = trial.suggest_int('batch_size', 16, 256)

        # Create sequences
        X_seq, y_seq = create_lstm_train_sequences(X, y, sequence_length)

        X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        input_size = X_train.shape[2]
        epochs = 1000
        patience = 10

        model = LSTMModel(input_size, hidden_size, num_blocks, num_layers, dropout_rate, classification=False).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_rmse = np.inf
        epochs_no_improve = 0

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

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
            val_rmse = 0
            with torch.no_grad():
                for input_val, target_val in val_loader:
                    input_val, target_val = input_val.to(device), target_val.to(device)
                    val_output = model(input_val)
                    val_rmse += root_mean_squared_error(target_val.cpu(), val_output.cpu()).item()

            val_rmse /= len(val_loader)

            # Report intermediate objective value
            trial.report(val_rmse, epoch)

            # Prune unpromising trials
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        return best_val_rmse

    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lstm_regression_objective, n_trials=MAX_TRIALS)

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
            with open(new_params_path, 'w') as f:
                json.dump(trial.params, f)
            trial_params = trial.params

            X_seq, y_seq = create_lstm_train_sequences(X, y, trial_params['sequence_length'])

            X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=TEST_SIZE,
                                                              random_state=RANDOM_STATE)

            input_size = X_train.shape[2]
            epochs = 1000
            patience = 10

            model = LSTMModel(input_size, trial_params['hidden_size'], trial_params['num_blocks'], trial_params['num_layers'], trial_params['dropout_rate'], classification=False).to(device)
            optimizer = optim.Adam(model.parameters(), lr=trial_params['lr'])
            criterion = nn.MSELoss()

            best_val_rmse = np.inf
            epochs_no_improve = 0

            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=trial_params['batch_size'], shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=trial_params['batch_size'], shuffle=False)

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
                val_rmse = 0
                with torch.no_grad():
                    for input_val, target_val in val_loader:
                        input_val, target_val = input_val.to(device), target_val.to(device)
                        val_output = model(input_val)
                        val_rmse += root_mean_squared_error(target_val.cpu(), val_output.cpu()).item()

                val_rmse /= len(val_loader)

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
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
    ticker_df.to_csv(Ticker_Hyperparams_Model_Metrics_Csv, index=False)

def lstm_regression_resume_training(X, y, gpu_available, ticker_symbol, hyperparameter_search=False, delete_old_data=False):

    if delete_old_data:
        delete_hyperparameter_search_model(ticker_symbol, Model_Type)

    all_existed = True
    for i in range(1, 6):
        hyperparameters_search_model_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.pth'
        hyperparameters_search_model_params_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.json'

        if not os.path.exists(hyperparameters_search_model_path) or not os.path.exists(hyperparameters_search_model_params_path):
            all_existed = False
            break

    if not all_existed or hyperparameter_search:
        lstm_regression_hyperparameters_search(X, y, gpu_available, ticker_symbol)

    for i in range(1, 6):
        hyperparameters_search_model_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.pth'
        trained_model_path = f'{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.pth'

        hyperparameters_search_model_params_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.json'
        trained_model_params_path = f'{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.json'

        shutil.copy2(hyperparameters_search_model_path, trained_model_path)
        shutil.copy2(hyperparameters_search_model_params_path, trained_model_params_path)

    hyperparameters_search_model_df = load_or_create_ticker_df(Ticker_Hyperparams_Model_Metrics_Csv)

    trained_model_df = load_or_create_ticker_df(Ticker_Trained_Model_Metrics_Csv)
    if ticker_symbol not in trained_model_df['Ticker_Symbol'].values:
        # Create a new DataFrame for the new row
        new_row = pd.DataFrame({'Ticker_Symbol': [ticker_symbol]})
        # Concatenate the new row to the existing DataFrame
        trained_model_df = pd.concat([trained_model_df, new_row], ignore_index=True)

    # List of columns to copy
    columns_to_copy = [f'{Model_Type}_1', f'{Model_Type}_2', f'{Model_Type}_3', f'{Model_Type}_4', f'{Model_Type}_5']

    # Copy the values from hyperparameters_search_model_df to trained_model_df for the specific row
    for column in columns_to_copy:
        trained_model_df.loc[trained_model_df['Ticker_Symbol'] == ticker_symbol, column] = \
        hyperparameters_search_model_df.loc[
            hyperparameters_search_model_df['Ticker_Symbol'] == ticker_symbol, column].values[0]

    trained_model_df.to_csv(Ticker_Trained_Model_Metrics_Csv, index=False)

def lstm_regression_predict(X, gpu_available, ticker_symbol, no=1):
    device = torch.device('cuda' if gpu_available and torch.cuda.is_available() else 'cpu')
    trained_model_path = f'{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{no}.pth'
    trained_model_params_path = f'{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{no}.json'

    # Check if the model exists
    if not os.path.exists(trained_model_path) or not os.path.exists(trained_model_params_path):
        print(f"Model or parameters file not found for {ticker_symbol} with number {no}.")
        return None

    # Load the model parameters from the JSON file
    with open(trained_model_params_path, 'r') as f:
        model_params = json.load(f)

    # Convert directly to tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    sequence_length = model_params['sequence_length']
    hidden_size = model_params['hidden_size']
    num_layers = model_params['num_layers']
    num_blocks = model_params['num_blocks']
    dropout_rate = model_params['dropout_rate']
    batch_size = model_params['batch_size']

    # Create sequences from X
    X_seq = create_lstm_predict_sequences(X, sequence_length)

    input_size = X_seq.shape[2]

    # Initialize the model with the loaded parameters and move it to the device
    model = LSTMModel(input_size, hidden_size, num_blocks, num_layers, dropout_rate, classification=False).to(device)
    model.load_state_dict(torch.load(trained_model_path, map_location=device, weights_only=True), strict=False)
    model.eval()  # Set the model to evaluation mode

    input_tensor = X_seq.to(device)

    # Create a DataLoader for batch processing
    data_loader = DataLoader(TensorDataset(input_tensor), batch_size=batch_size, shuffle=False)

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
