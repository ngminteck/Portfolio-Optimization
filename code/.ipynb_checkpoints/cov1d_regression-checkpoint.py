import torch
import optuna
import json
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

from directory_manager import *
from cov1d import *


def conv1d_regression_hyperparameters_search(X, y, gpu_available, ticker_symbol, delete_old_data=False):
    if delete_old_data:
        delete_hyperparameter_search_model(ticker_symbol, "conv1d-regression")

    model_path = f'../models/hyperparameters-search-models/pytorch/conv1d-regression/{ticker_symbol}.pth'
    csv_path = f'../models/hyperparameters-search-models/ticker-all-models-best-hyperparameters-list.csv'
    params_path = f'../models/best-hyperparameters/pytorch/conv1d-regression/{ticker_symbol}.json'

    device = torch.device('cuda' if gpu_available and torch.cuda.is_available() else 'cpu')

    # Convert DataFrame to numpy array
    X = X.to_numpy()
    y = y.to_numpy().reshape(-1, 1)

    # Reshape X for Conv1D
    NUM_CHANNELS = 1
    X = X.reshape((X.shape[0], NUM_CHANNELS, -1))  # Reshape for Conv1D: (batch_size, num_channels, sequence_length)

    # Split data into training and validation sets
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def conv1d_objective(trial):
        in_channels = X_train.shape[1]
        out_channels = trial.suggest_int('out_channels', 16, 128)
        kernel_size = trial.suggest_int('kernel_size', 3, 7)
        num_blocks = trial.suggest_int('num_blocks', 1, 5)
        l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-2)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        epochs = trial.suggest_int('epochs', 10, 100)

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

    study = optuna.create_study(direction='minimize')
    study.optimize(conv1d_objective, n_trials=100)

    best_model = Conv1DModel(X.shape[1], study.best_params['out_channels'], study.best_params['kernel_size'],
                             study.best_params['num_blocks'], study.best_params['l2_lambda'],
                             study.best_params['dropout_rate'], classification=False).to(device)

    ticker_df = load_or_create_ticker_df(csv_path)

    # Update ticker_df and save the best model
    metric_col = 'Best_Cov1D_Regression_RMSE'
    path_col = 'Best_Cov1D_Regression_Path'

    if ticker_symbol in ticker_df['Ticker_Symbol'].values:
        current_score = ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, metric_col].values[0]
        if pd.isnull(current_score) or study.best_value < current_score:
            ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, [metric_col, path_col]] = [study.best_value,
                                                                                                  model_path]
            torch.save(best_model.state_dict(), model_path)
            with open(params_path, 'w') as f:
                json.dump(study.best_params, f)
            print(f"parameters for {ticker_symbol} saved to {params_path}")
            ticker_df.to_csv(csv_path, index=False)
            print(f"Best model for {ticker_symbol} saved with RMSE: {study.best_value}")
        else:
            print(f"Previous model RMSE: {current_score} is better for {ticker_symbol} than RMSE: {study.best_value}")
    else:
        new_row = pd.DataFrame(
            {'Ticker_Symbol': [ticker_symbol], metric_col: [study.best_value], path_col: [model_path]})
        ticker_df = pd.concat([ticker_df, new_row], ignore_index=True)
        torch.save(best_model.state_dict(), model_path)
        with open(params_path, 'w') as f:
            json.dump(study.best_params, f)
        print(f"parameters for {ticker_symbol} saved to {params_path}")
        ticker_df.to_csv(csv_path, index=False)
        print(f"Best model for {ticker_symbol} saved with RMSE: {study.best_value}")