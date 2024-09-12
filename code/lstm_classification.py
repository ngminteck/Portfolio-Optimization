import torch
import optuna
import json
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from directory_manager import *
from lstm import *


def lstm_classification_hyperparameters_search(X, y, gpu_available, ticker_symbol, delete_old_data=False):
    if delete_old_data:
        delete_hyperparameter_search_model(ticker_symbol, "lstm-classification")

    model_path = f'../models/hyperparameters-search-models/pytorch/lstm-classification/{ticker_symbol}.pth'
    csv_path = f'../models/hyperparameters-search-models/ticker-all-models-best-hyperparameters-list.csv'
    params_path = f'../models/best-hyperparameters/pytorch/lstm-classification/{ticker_symbol}.json'

    device = torch.device('cuda' if gpu_available and torch.cuda.is_available() else 'cpu')

    # Convert DataFrame to numpy array
    X = X.to_numpy()
    y = y.to_numpy()

    def create_sequences(X, y, sequence_length):
        sequences_X, sequences_y = [], []
        for i in range(len(X) - sequence_length + 1):
            sequences_X.append(X[i:i + sequence_length])
            sequences_y.append(y[i + sequence_length - 1])
        return np.array(sequences_X), np.array(sequences_y)

    def lstm_objective(trial):
        sequence_length = trial.suggest_int('sequence_length',2, 30)

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

    study = optuna.create_study(direction='maximize')
    study.optimize(lstm_objective, n_trials=100)

    # Reshape X to match the best sequence length
    best_sequence_length = study.best_params['sequence_length']
    X_seq, y_seq = create_sequences(X, y, best_sequence_length)

    best_model = LSTMModel(X_seq.shape[2], study.best_params['hidden_size'], study.best_params['num_blocks'],
                           study.best_params['num_layers'], study.best_params['dropout_rate'], classification=True).to(
        device)

    ticker_df = load_or_create_ticker_df(csv_path)

    # Update ticker_df and save the best model
    metric_col = 'Best_LSTM_Classification_Accuracy'
    path_col = 'Best_LSTM_Classification_Path'

    if ticker_symbol in ticker_df['Ticker_Symbol'].values:
        current_score = ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, metric_col].values[0]
        if pd.isnull(current_score) or study.best_value > current_score:
            ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, [metric_col, path_col]] = [study.best_value,
                                                                                                  model_path]
            torch.save(best_model.state_dict(), model_path)
            with open(params_path, 'w') as f:
                json.dump(study.best_params, f)
            print(f"parameters for {ticker_symbol} saved to {params_path}")
            ticker_df.to_csv(csv_path, index=False)
            print(f"Best model for {ticker_symbol} saved with accuracy: {study.best_value}")
        else:
            print(
                f"Previous model accuracy: {current_score} is better for {ticker_symbol} than accuracy: {study.best_value}")
    else:
        new_row = pd.DataFrame(
            {'Ticker_Symbol': [ticker_symbol], metric_col: [study.best_value], path_col: [model_path]})
        ticker_df = pd.concat([ticker_df, new_row], ignore_index=True)
        torch.save(best_model.state_dict(), model_path)
        with open(params_path, 'w') as f:
            json.dump(study.best_params, f)
        print(f"parameters for {ticker_symbol} saved to {params_path}")
        ticker_df.to_csv(csv_path, index=False)
        print(f"Best model for {ticker_symbol} saved with accuracy: {study.best_value}")