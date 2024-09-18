import xgboost

from data_preprocessing import *

from cov1d_classification import *
from cov1d_regression import *
from lstm_classification import *
from lstm_regression import *
from cov1d_lstm_classification import *
from cov1d_lstm_regression import *

from xgbrfclassifier import *
from xgbrfregressor import *
from xgbclassifier_gbtree import *
from xgbregressor_gbtree import *


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

    path = '../data/train'
    ticker_list = []
    if os.path.exists(path):
        ticker_list = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.csv')]



    for ticker_symbol in ticker_list:

        original_df = pd.read_csv(f"../data/all/{ticker_symbol}.csv")

        # Create new columns in a separate DataFrame
        new_columns = pd.DataFrame(index=original_df.index)

        new_columns['Top_1_CNN1D_Diff_Predict'] = np.nan
        new_columns['Top_1_CNN1D_Price_Predict'] = np.nan
        new_columns['Top_5_CNN1D_Diff_Predict'] = np.nan
        new_columns['Top_5_CNN1D_Price_Predict'] = np.nan

        new_columns['Top_1_LSTM_Diff_Predict'] = np.nan
        new_columns['Top_1_LSTM_Price_Predict'] = np.nan
        new_columns['Top_5_LSTM_Diff_Predict'] = np.nan
        new_columns['Top_5_LSTM_Price_Predict'] = np.nan

        new_columns['Top_1_CNN1D_LSTM_Diff_Predict'] = np.nan
        new_columns['Top_1_CNN1D_LSTM_Price_Predict'] = np.nan
        new_columns['Top_5_CNN1D_LSTM_Diff_Predict'] = np.nan
        new_columns['Top_5_CNN1D_LSTM_Price_Predict'] = np.nan

        new_columns['Top_1_RF_Diff_Predict'] = np.nan
        new_columns['Top_1_RF_Price_Predict'] = np.nan
        new_columns['Top_5_RF_Diff_Predict'] = np.nan
        new_columns['Top_5_RF_Price_Predict'] = np.nan

        new_columns['Top_1_GT_Diff_Predict'] = np.nan
        new_columns['Top_1_GT_Price_Predict'] = np.nan
        new_columns['Top_5_GT_Diff_Predict'] = np.nan
        new_columns['Top_5_GT_Price_Predict'] = np.nan

        new_columns['Top_1_Combined_Diff_Predict'] = np.nan
        new_columns['Top_1_Combined_Price_Predict'] = np.nan
        new_columns['Top_5_Combined_Diff_Predict'] = np.nan
        new_columns['Top_5_Combined_Price_Predict'] = np.nan

        conv1d_regression_df = []
        lstm_regression_df = []
        conv1d_lstm_regression_df = []
        xgbrfregressor_regression_df = []
        xgbregressor_regression_df = []

        metric_df = pd.read_csv(f"../data/all/{ticker_symbol}.csv")

        X, metric_df = predict_preprocess_data(metric_df)

        for i in range(1, 6):
            result = conv1d_regression_predict(X, gpu_available, ticker_symbol, i)
            conv1d_regression_df.append(result)

            result = lstm_regression_predict(X, gpu_available, ticker_symbol, i)
            lstm_regression_df.append(result)

            result = conv1d_lstm_regression_predict(X, gpu_available, ticker_symbol, i)
            conv1d_lstm_regression_df.append(result)

            result = xgbrfregressor_predict(X, ticker_symbol, i)
            xgbrfregressor_regression_df.append(result)

            result = xgbregressor_gbtree_predict(X, ticker_symbol, i)
            xgbregressor_regression_df.append(result)

        print(f"All model for {ticker_symbol} predicted successfully.")

        # Combine all DataFrame lists into one list
        all_dfs = (conv1d_regression_df + lstm_regression_df +
                   conv1d_lstm_regression_df + xgbrfregressor_regression_df +
                   xgbregressor_regression_df)

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

        # Assigning values to 'Top_1_CNN1D_Diff_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_1_CNN1D_Diff_Predict'] = conv1d_regression_df[0].iloc[:,
                                                                                     0].values

        # Calculating and assigning 'Top_5_CNN1D_Diff_Predict'
        values_list = [df.iloc[:, 0].values for df in conv1d_regression_df]
        abs_mean = np.mean(np.abs(values_list), axis=0)
        signs = np.sign(values_list)
        majority_sign = np.sign(np.sum(signs, axis=0))
        new_columns.loc[new_columns.index[-min_rows:], 'Top_5_CNN1D_Diff_Predict'] = abs_mean * majority_sign

        # Assigning values to 'Top_1_LSTM_Diff_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_1_LSTM_Diff_Predict'] = lstm_regression_df[0].iloc[:,
                                                                                    0].values

        # Calculating and assigning 'Top_5_LSTM_Diff_Predict'
        values_list = [df.iloc[:, 0].values for df in lstm_regression_df]
        abs_mean = np.mean(np.abs(values_list), axis=0)
        signs = np.sign(values_list)
        majority_sign = np.sign(np.sum(signs, axis=0))
        new_columns.loc[new_columns.index[-min_rows:], 'Top_5_LSTM_Diff_Predict'] = abs_mean * majority_sign

        # Assigning values to 'Top_1_CNN1D_LSTM_Diff_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_1_CNN1D_LSTM_Diff_Predict'] = conv1d_lstm_regression_df[
                                                                                              0].iloc[:, 0].values

        # Calculating and assigning 'Top_5_CNN1D_LSTM_Diff_Predict'
        values_list = [df.iloc[:, 0].values for df in conv1d_lstm_regression_df]
        abs_mean = np.mean(np.abs(values_list), axis=0)
        signs = np.sign(values_list)
        majority_sign = np.sign(np.sum(signs, axis=0))
        new_columns.loc[new_columns.index[-min_rows:], 'Top_5_CNN1D_LSTM_Diff_Predict'] = abs_mean * majority_sign

        # Assigning values to 'Top_1_RF_Diff_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_1_RF_Diff_Predict'] = xgbrfregressor_regression_df[0].iloc[
                                                                                  :, 0].values

        # Calculating and assigning 'Top_5_RF_Diff_Predict'
        values_list = [df.iloc[:, 0].values for df in xgbrfregressor_regression_df]
        abs_mean = np.mean(np.abs(values_list), axis=0)
        signs = np.sign(values_list)
        majority_sign = np.sign(np.sum(signs, axis=0))
        new_columns.loc[new_columns.index[-min_rows:], 'Top_5_RF_Diff_Predict'] = abs_mean * majority_sign

        # Assigning values to 'Top_1_GT_Diff_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_1_GT_Diff_Predict'] = xgbregressor_regression_df[0].iloc[:,
                                                                                  0].values

        # Calculating and assigning 'Top_5_GT_Diff_Predict'
        values_list = [df.iloc[:, 0].values for df in xgbregressor_regression_df]
        abs_mean = np.mean(np.abs(values_list), axis=0)
        signs = np.sign(values_list)
        majority_sign = np.sign(np.sum(signs, axis=0))
        new_columns.loc[new_columns.index[-min_rows:], 'Top_5_GT_Diff_Predict'] = abs_mean * majority_sign

        # Calculating and assigning 'Top_1_Combined_Diff_Predict'
        values_list = [df.iloc[:, 0].values for df in [conv1d_regression_df[0], lstm_regression_df[0],
                                                       conv1d_lstm_regression_df[0], xgbrfregressor_regression_df[0],
                                                       xgbregressor_regression_df[0]]]
        abs_mean = np.mean(np.abs(values_list), axis=0)
        signs = np.sign(values_list)
        majority_sign = np.sign(np.sum(signs, axis=0))
        new_columns.loc[new_columns.index[-min_rows:], 'Top_1_Combined_Diff_Predict'] = abs_mean * majority_sign

        # Calculating and assigning 'Top_5_Combined_Diff_Predict'
        # Concatenate all columns from each DataFrame
        all_dfs = (conv1d_regression_df + lstm_regression_df +
                   conv1d_lstm_regression_df + xgbrfregressor_regression_df +
                   xgbregressor_regression_df)

        values_list = [df.iloc[:, 0].values for df in all_dfs]
        abs_mean = np.mean(np.abs(values_list), axis=0)
        signs = np.sign(values_list)
        majority_sign = np.sign(np.sum(signs, axis=0))

        # Assign the calculated values to 'Top_5_Combined_Diff_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_5_Combined_Diff_Predict'] = abs_mean * majority_sign

        # Assigning values to 'Top_1_CNN1D_Price_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_1_CNN1D_Price_Predict'] = (
                original_df['DAILY_MIDPRICE'].iloc[-min_rows:].values + new_columns['Top_1_CNN1D_Diff_Predict'].iloc[
                                                                        -min_rows:].values
        )

        # Assigning values to 'Top_5_CNN1D_Price_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_5_CNN1D_Price_Predict'] = (
                original_df['DAILY_MIDPRICE'].iloc[-min_rows:].values + new_columns['Top_5_CNN1D_Diff_Predict'].iloc[
                                                                        -min_rows:].values
        )

        # Assigning values to 'Top_1_LSTM_Price_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_1_LSTM_Price_Predict'] = (
                original_df['DAILY_MIDPRICE'].iloc[-min_rows:].values + new_columns['Top_1_LSTM_Diff_Predict'].iloc[
                                                                        -min_rows:].values
        )

        # Assigning values to 'Top_5_LSTM_Price_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_5_LSTM_Price_Predict'] = (
                original_df['DAILY_MIDPRICE'].iloc[-min_rows:].values + new_columns['Top_5_LSTM_Diff_Predict'].iloc[
                                                                        -min_rows:].values
        )

        # Assigning values to 'Top_1_CNN1D_LSTM_Price_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_1_CNN1D_LSTM_Price_Predict'] = (
                original_df['DAILY_MIDPRICE'].iloc[-min_rows:].values + new_columns[
                                                                            'Top_1_CNN1D_LSTM_Diff_Predict'].iloc[
                                                                        -min_rows:].values
        )

        # Assigning values to 'Top_5_CNN1D_LSTM_Price_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_5_CNN1D_LSTM_Price_Predict'] = (
                original_df['DAILY_MIDPRICE'].iloc[-min_rows:].values + new_columns[
                                                                            'Top_5_CNN1D_LSTM_Diff_Predict'].iloc[
                                                                        -min_rows:].values
        )

        # Assigning values to 'Top_1_RF_Price_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_1_RF_Price_Predict'] = (
                original_df['DAILY_MIDPRICE'].iloc[-min_rows:].values + new_columns['Top_1_RF_Diff_Predict'].iloc[
                                                                        -min_rows:].values
        )

        # Assigning values to 'Top_5_RF_Price_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_5_RF_Price_Predict'] = (
                original_df['DAILY_MIDPRICE'].iloc[-min_rows:].values + new_columns['Top_5_RF_Diff_Predict'].iloc[
                                                                        -min_rows:].values
        )

        # Assigning values to 'Top_1_GT_Price_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_1_GT_Price_Predict'] = (
                original_df['DAILY_MIDPRICE'].iloc[-min_rows:].values + new_columns['Top_1_GT_Diff_Predict'].iloc[
                                                                        -min_rows:].values
        )

        # Assigning values to 'Top_5_GT_Price_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_5_GT_Price_Predict'] = (
                original_df['DAILY_MIDPRICE'].iloc[-min_rows:].values + new_columns['Top_5_GT_Diff_Predict'].iloc[
                                                                        -min_rows:].values
        )

        # Assigning values to 'Top_1_Combined_Price_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_1_Combined_Price_Predict'] = (
                original_df['DAILY_MIDPRICE'].iloc[-min_rows:].values + new_columns['Top_1_Combined_Diff_Predict'].iloc[
                                                                        -min_rows:].values
        )

        # Assigning values to 'Top_5_Combined_Price_Predict'
        new_columns.loc[new_columns.index[-min_rows:], 'Top_5_Combined_Price_Predict'] = (
                original_df['DAILY_MIDPRICE'].iloc[-min_rows:].values + new_columns['Top_5_Combined_Diff_Predict'].iloc[
                                                                        -min_rows:].values
        )

        original_df = pd.concat([original_df, new_columns], axis=1)

        SLICE_SIZE = 87 + 30
        original_df = original_df.iloc[SLICE_SIZE:, :]

        original_df.to_csv(f'../result/ticker/{ticker_symbol}.csv')
        original_df = original_df.iloc[:-1, :]

        actual_signs = np.sign(original_df['DAILY_MIDPRICE_CHANGE'].diff().fillna(0))
        actual_values = original_df['DAILY_MIDPRICE_CHANGE']
        actual_values_mean = actual_values.mean()

        ticker_df = load_or_create_ticker_metric_df('../result/ticker_metrics.csv')
        if ticker_symbol not in ticker_df['Ticker_Symbol'].values:
            # Create a new DataFrame for the new row
            new_row = pd.DataFrame({'Ticker_Symbol': [ticker_symbol]})
            # Concatenate the new row to the existing DataFrame
            ticker_df = pd.concat([ticker_df, new_row], ignore_index=True)

        predicted_signs = np.sign(original_df['Top_1_CNN1D_Diff_Predict'])
        sign_accuracy = (actual_signs == predicted_signs).mean() * 100
        predicted_values = original_df['Top_1_CNN1D_Diff_Predict']
        diff = predicted_values.mean() - actual_values_mean
        percentage_rmse = (diff / actual_values_mean) * 100
        # Update the DataFrame with the calculated values
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_1_CNN1D_Sign_Accuracy_PERCENT'] = sign_accuracy
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_1_CNN1D_Value_RMSE_PERCENT'] = percentage_rmse

        predicted_signs = np.sign(original_df['Top_5_CNN1D_Diff_Predict'])
        sign_accuracy = (actual_signs == predicted_signs).mean() * 100
        predicted_values = original_df['Top_5_CNN1D_Diff_Predict']
        diff = predicted_values.mean() - actual_values_mean
        percentage_rmse = (diff / actual_values_mean) * 100
        # Update the DataFrame with the calculated values
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_5_CNN1D_Sign_Accuracy_PERCENT'] = sign_accuracy
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_5_CNN1D_Value_RMSE_PERCENT'] = percentage_rmse

        predicted_signs = np.sign(original_df['Top_1_LSTM_Diff_Predict'])
        sign_accuracy = (actual_signs == predicted_signs).mean() * 100
        predicted_values = original_df['Top_1_LSTM_Diff_Predict']
        diff = predicted_values.mean() - actual_values_mean
        percentage_rmse = (diff / actual_values_mean) * 100
        # Update the DataFrame with the calculated values
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_1_LSTM_Sign_Accuracy_PERCENT'] = sign_accuracy
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_1_LSTM_Value_RMSE_PERCENT'] = percentage_rmse

        predicted_signs = np.sign(original_df['Top_5_LSTM_Diff_Predict'])
        sign_accuracy = (actual_signs == predicted_signs).mean() * 100
        predicted_values = original_df['Top_5_LSTM_Diff_Predict']
        diff = predicted_values.mean() - actual_values_mean
        percentage_rmse = (diff / actual_values_mean) * 100
        # Update the DataFrame with the calculated values
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_5_LSTM_Sign_Accuracy_PERCENT'] = sign_accuracy
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_5_LSTM_Value_RMSE_PERCENT'] = percentage_rmse

        predicted_signs = np.sign(original_df['Top_1_CNN1D_LSTM_Diff_Predict'])
        sign_accuracy = (actual_signs == predicted_signs).mean() * 100
        predicted_values = original_df['Top_1_CNN1D_LSTM_Diff_Predict']
        diff = predicted_values.mean() - actual_values_mean
        percentage_rmse = (diff / actual_values_mean) * 100
        # Update the DataFrame with the calculated values
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_1_CNN1D_LSTM_Sign_Accuracy_PERCENT'] = sign_accuracy
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_1_CNN1D_LSTM_Value_RMSE_PERCENT'] = percentage_rmse

        predicted_signs = np.sign(original_df['Top_5_CNN1D_LSTM_Diff_Predict'])
        sign_accuracy = (actual_signs == predicted_signs).mean() * 100
        predicted_values = original_df['Top_5_CNN1D_LSTM_Diff_Predict']
        diff = predicted_values.mean() - actual_values_mean
        percentage_rmse = (diff / actual_values_mean) * 100
        # Update the DataFrame with the calculated values
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_5_CNN1D_LSTM_Sign_Accuracy_PERCENT'] = sign_accuracy
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_5_CNN1D_LSTM_Value_RMSE_PERCENT'] = percentage_rmse

        predicted_signs = np.sign(original_df['Top_1_RF_Diff_Predict'])
        sign_accuracy = (actual_signs == predicted_signs).mean() * 100
        predicted_values = original_df['Top_1_RF_Diff_Predict']
        diff = predicted_values.mean() - actual_values_mean
        percentage_rmse = (diff / actual_values_mean) * 100
        # Update the DataFrame with the calculated values
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_1_RF_Sign_Accuracy_PERCENT'] = sign_accuracy
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_1_RF_Value_RMSE_PERCENT'] = percentage_rmse

        predicted_signs = np.sign(original_df['Top_5_RF_Diff_Predict'])
        sign_accuracy = (actual_signs == predicted_signs).mean() * 100
        predicted_values = original_df['Top_5_RF_Diff_Predict']
        diff = predicted_values.mean() - actual_values_mean
        percentage_rmse = (diff / actual_values_mean) * 100
        # Update the DataFrame with the calculated values
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_5_RF_Sign_Accuracy_PERCENT'] = sign_accuracy
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_5_RF_Value_RMSE_PERCENT'] = percentage_rmse

        predicted_signs = np.sign(original_df['Top_1_GT_Diff_Predict'])
        sign_accuracy = (actual_signs == predicted_signs).mean() * 100
        predicted_values = original_df['Top_1_GT_Diff_Predict']
        diff = predicted_values.mean() - actual_values_mean
        percentage_rmse = (diff / actual_values_mean) * 100
        # Update the DataFrame with the calculated values
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_1_GT_Sign_Accuracy_PERCENT'] = sign_accuracy
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_1_GT_Value_RMSE_PERCENT'] = percentage_rmse

        predicted_signs = np.sign(original_df['Top_5_GT_Diff_Predict'])
        sign_accuracy = (actual_signs == predicted_signs).mean() * 100
        predicted_values = original_df['Top_5_GT_Diff_Predict']
        diff = predicted_values.mean() - actual_values_mean
        percentage_rmse = (diff / actual_values_mean) * 100
        # Update the DataFrame with the calculated values
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_5_GT_Sign_Accuracy_PERCENT'] = sign_accuracy
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_5_GT_Value_RMSE_PERCENT'] = percentage_rmse

        predicted_signs = np.sign(original_df['Top_1_Combined_Diff_Predict'])
        sign_accuracy = (actual_signs == predicted_signs).mean() * 100
        predicted_values = original_df['Top_1_Combined_Diff_Predict']
        diff = predicted_values.mean() - actual_values_mean
        percentage_rmse = (diff / actual_values_mean) * 100
        # Update the DataFrame with the calculated values
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_1_Combined_Sign_Accuracy_PERCENT'] = sign_accuracy
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_1_Combined_Value_RMSE_PERCENT'] = percentage_rmse

        predicted_signs = np.sign(original_df['Top_5_Combined_Diff_Predict'])
        sign_accuracy = (actual_signs == predicted_signs).mean() * 100
        predicted_values = original_df['Top_5_Combined_Diff_Predict']
        diff = predicted_values.mean() - actual_values_mean
        percentage_rmse = (diff / actual_values_mean) * 100
        # Update the DataFrame with the calculated values
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_5_Combined_Sign_Accuracy_PERCENT'] = sign_accuracy
        ticker_df.loc[ticker_df['Ticker_Symbol'] == ticker_symbol, 'Top_5_Combined_Value_RMSE_PERCENT'] = percentage_rmse

        ticker_df.to_csv('../result/ticker_metrics.csv', index=False)

        print(f"{ticker_symbol} done evaluate.")




