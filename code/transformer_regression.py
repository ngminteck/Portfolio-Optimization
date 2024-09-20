import optuna
import json
import torch.optim as optim
from torch.utils.data import DataLoader
import shutil

from directory_manager import *
from optuna_config import *
from transformer import *
from sequence_length import *

Model_Type = "transformer_regression"

def transformer_regression_hyperparameters_search(X, y, gpu_available, ticker_symbol):
    device = torch.device('cuda' if gpu_available and torch.cuda.is_available() else 'cpu')

    # Convert DataFrame to tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)

    # Split data into training and validation sets
    train_size = int(0.8 * len(X_tensor))
    val_size = len(X_tensor) - train_size
    input_train, input_val = torch.utils.data.random_split(X_tensor, [train_size, val_size])
    target_train, target_val = torch.utils.data.random_split(y_tensor, [train_size, val_size])

    def transformer_regression_objective(trial):
        num_heads = trial.suggest_int('num_heads', 2, 8)
        num_layers = trial.suggest_int('num_layers', 2, 6)
        dropout = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-5, 1e-1)
        epochs = 1000
        patience = 10

        input_dim = X_tensor.shape[1]
        embed_dim = ((input_dim + num_heads - 1) // num_heads) * num_heads

        model = TransformerModel(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, output_dim=1, dropout=dropout, is_classification=False).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_rmse = np.inf
        epochs_no_improve = 0

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
                val_rmse = torch.sqrt(criterion(val_output, target_val)).item()

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
    study.optimize(transformer_regression_objective,  n_trials=MAX_TRIALS)

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

            num_heads = trial_params['num_heads']
            num_layers = trial_params['num_layers']
            dropout = trial_params['dropout_rate']
            lr = trial_params['lr']
            epochs = 1000
            patience = 10

            input_dim = X_tensor.shape[1]
            embed_dim = ((input_dim + num_heads - 1) // num_heads) * num_heads

            model = TransformerModel(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads,
                                     num_layers=num_layers, output_dim=1, dropout=dropout, is_classification=False).to(
                device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            best_val_rmse = np.inf
            epochs_no_improve = 0

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
                    val_rmse = torch.sqrt(criterion(val_output, target_val)).item()

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

def transformer_regression_resume_training(X, y, gpu_available, ticker_symbol, hyperparameter_search=False,
                                          delete_old_data=False):

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
        transformer_regression_hyperparameters_search(X, y, gpu_available, ticker_symbol)

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
    columns_to_copy = [f'{Model_Type}_1', f'{Model_Type}_2', f'{Model_Type}_3', f'{Model_Type}_4',
                       f'{Model_Type}_5']

    # Copy the values from hyperparameters_search_model_df to trained_model_df for the specific row
    for column in columns_to_copy:
        trained_model_df.loc[trained_model_df['Ticker_Symbol'] == ticker_symbol, column] = \
            hyperparameters_search_model_df.loc[
                hyperparameters_search_model_df['Ticker_Symbol'] == ticker_symbol, column].values[0]

    trained_model_df.to_csv(Ticker_Trained_Model_Metrics_Csv, index=False)




