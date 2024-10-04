import xgboost

from data_preprocessing import *

from xgbrfregressor import *
from xgbregressor_gbtree import *
from grnn_regression import *
from cov1d_regression import *
from lstm_regression import *
from cov1d_lstm_regression import *
from directory_manager import *
from metric import *

def main_evaluate():
    def is_gpu_available():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    gpu_available = is_gpu_available()
    print(f"GPU available: {gpu_available}")
    print(xgboost.build_info())

    path = '../data/all'
    ticker_list = []
    if os.path.exists(path):
        ticker_list = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.csv')]

    for ticker_symbol in ticker_list:

        xgbrfregressor_regression_df = []
        xgbregressor_regression_df = []
        grnn_regression_df = []
        conv1d_regression_df = []
        lstm_regression_df = []
        conv1d_lstm_regression_df = []
        transformer_regression_df = []

        X, y, y_scaler = predict_preprocess_data(ticker_symbol)

        predict_error = False

        for i in range(1, 6):
            result = xgbrfregressor_predict(X, ticker_symbol, i)
            if result is None:
                predict_error = True
                print(f"{ticker_symbol} encounter prediction error in RF")
                break
            inverse_transformed = y_scaler.inverse_transform(result.iloc[:, [0]])
            result.iloc[:, 0] = inverse_transformed[:, 0].astype('float32')
            xgbrfregressor_regression_df.append(result)

            result = xgbregressor_gbtree_predict(X, ticker_symbol, i)
            if result is None:
                predict_error = True
                print(f"{ticker_symbol} encounter prediction error in GT")
                break
            inverse_transformed = y_scaler.inverse_transform(result.iloc[:, [0]])
            result.iloc[:, 0] = inverse_transformed[:, 0].astype('float32')
            xgbregressor_regression_df.append(result)

            result = grnn_regression_predict(X, gpu_available, ticker_symbol, i)
            if result is None:
                predict_error = True
                print(f"{ticker_symbol} encounter prediction error in GRNN")
                break
            inverse_transformed = y_scaler.inverse_transform(result.iloc[:, [0]])
            result.iloc[:, 0] = inverse_transformed[:, 0].astype('float32')
            grnn_regression_df.append(result)

            result = conv1d_regression_predict(X, gpu_available, ticker_symbol, i)
            if result is None:
                predict_error = True
                print(f"{ticker_symbol} encounter prediction error in CNN1D")
                break
            inverse_transformed = y_scaler.inverse_transform(result.iloc[:, [0]])
            result.iloc[:, 0] = inverse_transformed[:, 0].astype('float32')
            conv1d_regression_df.append(result)

            result = lstm_regression_predict(X, gpu_available, ticker_symbol, i)
            if result is None:
                predict_error = True
                print(f"{ticker_symbol} encounter prediction error in LSTM")
                break
            inverse_transformed = y_scaler.inverse_transform(result.iloc[:, [0]])
            result.iloc[:, 0] = inverse_transformed[:, 0].astype('float32')
            lstm_regression_df.append(result)

            result = conv1d_lstm_regression_predict(X, gpu_available, ticker_symbol, i)
            if result is None:
                predict_error = True
                print(f"{ticker_symbol} encounter prediction error in CNN LSTM")
                break
            inverse_transformed = y_scaler.inverse_transform(result.iloc[:, [0]])
            result.iloc[:, 0] = inverse_transformed[:, 0].astype('float32')
            conv1d_lstm_regression_df.append(result)

        if predict_error:
            continue

        print(f"All model for {ticker_symbol} predicted successfully.")

        # Combine all DataFrame lists into one list
        all_dfs = (conv1d_regression_df + lstm_regression_df +
                   conv1d_lstm_regression_df + xgbrfregressor_regression_df +
                   xgbregressor_regression_df + grnn_regression_df)

        # Find the DataFrame with the minimum number of rows
        min_rows_df = min(all_dfs, key=lambda df: len(df))

        # Get the number of rows in the DataFrame with the minimum rows
        min_rows = len(min_rows_df)

        # Trim each DataFrame to have the same number of rows
        conv1d_regression_df = [df.iloc[-min_rows:] for df in conv1d_regression_df]
        lstm_regression_df = [df.iloc[-min_rows:] for df in lstm_regression_df]
        conv1d_lstm_regression_df = [df.iloc[-min_rows:] for df in conv1d_lstm_regression_df]
        xgbrfregressor_regression_df = [df.iloc[-min_rows:] for df in xgbrfregressor_regression_df]
        xgbregressor_regression_df = [df.iloc[-min_rows:] for df in xgbregressor_regression_df]
        grnn_regression_df = [df.iloc[-min_rows:] for df in grnn_regression_df]

        ticker_df = load_or_create_ticker_metric_df('../predicted_output/ticker_metrics.csv')
        if ticker_symbol not in ticker_df['Ticker_Symbol'].values:
            # Create a new DataFrame for the new row
            new_row = pd.DataFrame({'Ticker_Symbol': [ticker_symbol]})
            # Concatenate the new row to the existing DataFrame
            ticker_df = pd.concat([ticker_df, new_row], ignore_index=True)

        df = pd.read_csv(f"../data/all/{ticker_symbol}.csv")

        if df.isna().sum().sum() > 0 or df.isin([float('inf'), float('-inf')]).sum().sum() > 0:
            df = df.replace([float('inf'), float('-inf')], np.nan).dropna()

        predict_df = df[['Date', 'DAILY_CLOSEPRICE_CHANGE']].copy(deep=True)
        y = y.iloc[-min_rows:]
        predict_df = predict_df.iloc[-min_rows:]
        are_columns_same = predict_df['DAILY_CLOSEPRICE_CHANGE'].equals(y['DAILY_CLOSEPRICE_CHANGE'])
        print(are_columns_same)
        # List of model prefixes
        models = ['RF', 'GBT', 'GRNN', 'CNN', 'LSTM', 'CNN_LSTM']

        # Add columns for each model
        for model in models:
            for i in range(1, 6):
                column_name = f'{model}_{i}_Predicted_Close_Price_Change'
                predict_df[column_name] = np.nan

        for i in range(1, 6):
            column_name = f"RF_{i}_Sign_Accuracy"
            ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = accuracy_np(
                predict_df['DAILY_CLOSEPRICE_CHANGE'], xgbrfregressor_regression_df[i - 1].iloc[:, 0].values)
            column_name = f"RF_{i}_Profit_Loss"
            ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = profit_loss_np(
                predict_df['DAILY_CLOSEPRICE_CHANGE'], xgbrfregressor_regression_df[i - 1].iloc[:, 0].values)
            column_name = f"RF_{i}_Predicted_Close_Price_Change"
            predict_df[column_name] = xgbrfregressor_regression_df[i - 1].iloc[:, 0].values

            column_name = f"GBT_{i}_Sign_Accuracy"
            ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = accuracy_np(
                predict_df['DAILY_CLOSEPRICE_CHANGE'], xgbregressor_regression_df[i - 1].iloc[:, 0].values)
            column_name = f"GBT_{i}_Profit_Loss"
            ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = profit_loss_np(
                predict_df['DAILY_CLOSEPRICE_CHANGE'], xgbregressor_regression_df[i - 1].iloc[:, 0].values)
            column_name = f"GBT_{i}_Predicted_Close_Price_Change"
            predict_df[column_name] = xgbregressor_regression_df[i - 1].iloc[:, 0].values

            column_name = f"GRNN_{i}_Sign_Accuracy"
            ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = accuracy_np(
                predict_df['DAILY_CLOSEPRICE_CHANGE'], grnn_regression_df[i - 1].iloc[:, 0].values)
            column_name = f"GRNN_{i}_Profit_Loss"
            ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = profit_loss_np(
                predict_df['DAILY_CLOSEPRICE_CHANGE'], grnn_regression_df[i - 1].iloc[:, 0].values)
            column_name = f"GRNN_{i}_Predicted_Close_Price_Change"
            predict_df[column_name] = grnn_regression_df[i - 1].iloc[:, 0].values

            column_name = f"CNN_{i}_Sign_Accuracy"
            ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = accuracy_np(
                predict_df['DAILY_CLOSEPRICE_CHANGE'], conv1d_regression_df[i - 1].iloc[:, 0].values)
            column_name = f"CNN_{i}_Profit_Loss"
            ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = profit_loss_np(
                predict_df['DAILY_CLOSEPRICE_CHANGE'], conv1d_regression_df[i - 1].iloc[:, 0].values)
            column_name = f"CNN_{i}_Predicted_Close_Price_Change"
            predict_df[column_name] = conv1d_regression_df[i - 1].iloc[:, 0].values

            column_name = f"LSTM_{i}_Sign_Accuracy"
            ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = accuracy_np(
                predict_df['DAILY_CLOSEPRICE_CHANGE'], lstm_regression_df[i - 1].iloc[:, 0].values)
            column_name = f"LSTM_{i}_Profit_Loss"
            ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = profit_loss_np(
                predict_df['DAILY_CLOSEPRICE_CHANGE'], lstm_regression_df[i - 1].iloc[:, 0].values)
            column_name = f"LSTM_{i}_Predicted_Close_Price_Change"
            predict_df[column_name] = lstm_regression_df[i - 1].iloc[:, 0].values

            column_name = f"CNN_LSTM_{i}_Sign_Accuracy"
            ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = accuracy_np(
                predict_df['DAILY_CLOSEPRICE_CHANGE'], conv1d_lstm_regression_df[i - 1].iloc[:, 0].values)
            column_name = f"CNN_LSTM_{i}_Profit_Loss"
            ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, column_name] = profit_loss_np(
                predict_df['DAILY_CLOSEPRICE_CHANGE'], conv1d_lstm_regression_df[i - 1].iloc[:, 0].values)
            column_name = f"CNN_LSTM_{i}_Predicted_Close_Price_Change"
            predict_df[column_name] = conv1d_lstm_regression_df[i - 1].iloc[:, 0].values

        predict_df.to_csv(f'../predicted_output/ticker/{ticker_symbol}.csv', index=False)
        ticker_df.to_csv(f'../predicted_output/ticker_metrics.csv', index=False)
        print(f"{ticker_symbol} done evaluate.")







