import json
import joblib
import optuna
from xgboost import XGBRFClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shutil

from directory_manager import *
from optuna_config import *

Model_Type = "xgbrfclassifier"

def xgbrfclassifier_hyperparameters_search(X, y, gpu_available, ticker_symbol):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    def xgbrfclassifier_objective(trial):
        param = {
            'verbosity': 0,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'device': 'cuda' if gpu_available else 'cpu',
            'use_label_encoder': False,
            'n_estimators': 200,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),  # Adjusting range
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),  # Adjusting range
            'num_parallel_tree': 10 * os.cpu_count()

        }

        model = XGBRFClassifier(**param)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        preds = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, preds)

        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(xgbrfclassifier_objective, n_trials=MAX_TRIALS)

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
        new_model_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{rank}.pkl'
        new_params_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{rank}.json'
        new_feature_path = f'{Feature_Importance_Folder}{Model_Type}/{ticker_symbol}_{rank}.csv'

        if key.startswith('old'):
            # Extract the index from the key using split method
            old_index = key.split('_')[1]
            old_model_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{old_index}.pkl'
            old_params_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{old_index}.json'
            old_feature_path = f'{Feature_Importance_Folder}{Model_Type}/{ticker_symbol}_{old_index}.csv'

            rename_and_overwrite(old_model_path, new_model_path)
            rename_and_overwrite(old_params_path, new_params_path)
            rename_and_overwrite(old_feature_path, new_feature_path)

        else:
            trial_index = int(key.split('_')[1])
            trial = all_trials[trial_index]
            with open(new_params_path, 'w') as f:
                json.dump(trial.params, f)
            trial_params = trial.params
            model = XGBRFClassifier(**trial_params)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            joblib.dump(model, new_model_path)

            feature_importance = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance'])
            feature_importance = feature_importance.sort_values(by='importance', ascending=False)
            feature_importance.to_csv(new_feature_path)

        # Update ticker_df with the new metrics
        column_name = f"{Model_Type}_{rank}"
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = value

    # Save the updated ticker_df back to the CSV
    ticker_df.to_csv(Ticker_Hyperparams_Model_Metrics_Csv, index=False)


def xgbrfclassifier_resume_training(X, y, gpu_available, ticker_symbol, hyperparameter_search=False, delete_old_data=False):

    if delete_old_data:
        delete_hyperparameter_search_model(ticker_symbol, Model_Type)

    all_existed = True
    for i in range(1, 6):
        hyperparameters_search_model_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.pkl'
        hyperparameters_search_model_params_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.json'

        if not os.path.exists(hyperparameters_search_model_path) or not os.path.exists(hyperparameters_search_model_params_path):
            all_existed = False
            break

    if not all_existed or hyperparameter_search:
        xgbrfclassifier_hyperparameters_search(X, y, gpu_available, ticker_symbol)

    for i in range(1, 6):
        hyperparameters_search_model_path = f'{Hyperparameters_Search_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.pkl'
        trained_model_path = f'{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{i}.pkl'

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


def xgbrfclassifier_predict(X, ticker_symbol, no=1):
    trained_model_path = f'{Trained_Models_Folder}{Model_Type}/{ticker_symbol}_{no}.pkl'

    # Check if the model exists
    if not os.path.exists(trained_model_path):
        return None

    # Load the trained model
    model = joblib.load(trained_model_path)

    # Make predictions
    predictions = model.predict(X)

    # Return predictions as a DataFrame
    return pd.DataFrame(predictions, columns=['Predicted'])