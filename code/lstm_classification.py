import torch
import optuna
import json
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shutil

from directory_manager import *
from optuna_config import *
from lstm import *

Model_Type = "lstm_classification"

def lstm_classification_hyperparameters_search(X, y, gpu_available, ticker_symbol):
    device = torch.device('cuda' if gpu_available and torch.cuda.is_available() else 'cpu')

    X = X.to_numpy()
    y = y.to_numpy()

    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    def lstm_classification_objective(trial):

        sequence_length = trial.suggest_int('sequence_length', 2, 30)
        hidden_size = trial.suggest_int('hidden_size', 16, 128)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        num_blocks = trial.suggest_int('num_blocks', 1, 5)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-5, 1e-1)

        X_seq, y_seq = create_sequences(X, y, sequence_length)

        X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        input_size = X_train.shape[2]
        epochs = 1000
        patience = 10

        model = LSTMModel(input_size, hidden_size, num_blocks, num_layers, dropout_rate, classification=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_val_accuracy = -np.inf
        epochs_no_improve = 0

        input_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        target_train = torch.tensor(y_train, dtype=torch.long).to(device)

        input_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        target_val = torch.tensor(y_val, dtype=torch.long).to(device)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(input_train)
            loss = criterion(output, target_train)
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model(input_val)
                val_pred = val_output.argmax(dim=1)
                val_accuracy = accuracy_score(target_val.cpu(), val_pred.cpu())

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break

        return best_val_accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(lstm_classification_objective, n_trials=MAX_TRIALS)

    # Get all trials
    all_trials = study.trials

    # Sort trials by their objective values in descending order
    sorted_trials = sorted(all_trials, key=lambda trial: trial.value, reverse=True)

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

    sorted_metrics = dict(sorted(metrics.items(), key=lambda item: item[1], reverse=True))
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

            X_seq, y_seq = create_sequences(X, y, trial_params['sequence_length'])

            X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=TEST_SIZE,
                                                              random_state=RANDOM_STATE)

            input_size = X_train.shape[2]
            epochs = 1000
            patience = 10

            model = LSTMModel(input_size,trial_params['hidden_size'], trial_params['num_blocks'], trial_params['num_layers'], trial_params['dropout_rate'], classification=True).to(device)
            optimizer = optim.Adam(model.parameters(), lr=trial_params['lr'])
            criterion = nn.CrossEntropyLoss()

            best_val_accuracy = -np.inf
            epochs_no_improve = 0

            input_train = torch.tensor(X_train, dtype=torch.float32).to(device)
            target_train = torch.tensor(y_train, dtype=torch.long).to(device)

            input_val = torch.tensor(X_val, dtype=torch.float32).to(device)
            target_val = torch.tensor(y_val, dtype=torch.long).to(device)

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                output = model(input_train)
                loss = criterion(output, target_train)
                loss.backward()
                optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_output = model(input_val)
                    val_pred = val_output.argmax(dim=1)
                    val_accuracy = accuracy_score(target_val.cpu(), val_pred.cpu())

                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= patience:
                        break

            # Save the new model from trial
            torch.save(model.state_dict(), new_model_path)
            with open(new_params_path, 'w') as f:
                json.dump(trial.params, f)

        # Update ticker_df with the new metrics
        column_name = f"{Model_Type}_{rank}"
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = value

    # Save the updated ticker_df back to the CSV
    ticker_df.to_csv(Ticker_Hyperparams_Model_Metrics_Csv, index=False)

def lstm_classification_resume_training(X, y, gpu_available, ticker_symbol, hyperparameter_search=False, delete_old_data=False):

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
        lstm_classification_hyperparameters_search(X, y, gpu_available, ticker_symbol)

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
    columns_to_copy = [f'{Model_Type}_1', f'{Model_Type}_2', f'{Model_Type}_3', f'{Model_Type}_4', f'{Model_Type}_5']

    # Copy the values from hyperparameters_search_model_df to trained_model_df for the specific row
    for column in columns_to_copy:
        trained_model_df.loc[trained_model_df['Ticker_Symbol'] == ticker_symbol, column] = \
        hyperparameters_search_model_df.loc[
            hyperparameters_search_model_df['Ticker_Symbol'] == ticker_symbol, column].values[0]

    trained_model_df.to_csv(Ticker_Trained_Model_Metrics_Csv, index=False)