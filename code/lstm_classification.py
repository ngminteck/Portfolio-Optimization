import torch
import optuna
import json
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from directory_manager import *
from lstm import *

Model_Type = "lstm_classification"

def lstm_classification_objective(X, y, gpu_available, trial):
    X = X.to_numpy()
    y = y.to_numpy()

    sequence_length = trial.suggest_int('sequence_length', 2, 30)

    # Create sequences
    X_seq, y_seq = create_sequences(X, y, sequence_length)

    # Split data into training, validation, and test sets
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_STATE = 42

    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=TEST_SIZE + VAL_SIZE,
                                                        random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
                                                    random_state=RANDOM_STATE)

    device = torch.device('cuda' if gpu_available and torch.cuda.is_available() else 'cpu')

    input_size = X_train.shape[2]  # Number of features
    hidden_size = trial.suggest_int('hidden_size', 16, 128)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    num_blocks = trial.suggest_int('num_blocks', 1, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    epochs = 1000

    model = LSTMModel(input_size, hidden_size, num_blocks, num_layers, dropout_rate, classification=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float('lr', 1e-5, 1e-2))
    criterion = nn.CrossEntropyLoss()

    patience = 10
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
                print(f"Early stopping at epoch {epoch}")
                break

    return best_val_accuracy

def lstm_classification_hyperparameters_search(X, y, gpu_available, ticker_symbol, delete_old_data=False):
    if delete_old_data:
        delete_hyperparameter_search_model(ticker_symbol, Model_Type)

    study = optuna.create_study(direction='maximize')
    study.optimize(lstm_classification_objective, X, y, gpu_available, n_trials=100)

    top_trials = study.trials_dataframe().sort_values(by='value', ascending=False).head(5)

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

    sorted_metrics = {k: v for k, v in sorted(metrics.items(), key=lambda item: item[1], reverse=True)}

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