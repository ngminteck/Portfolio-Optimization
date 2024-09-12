import xgboost
import matplotlib.pyplot as plt

from directory_manager import *
from yahoo_finance import *
from technical_indicator import *
from news_sentiment_analysis import *
from data_preparation import *

from training_preprocessing import *
from testing_preprocessing import *
from xbclassifier import *
from xbregressor import *


logical_cores = os.cpu_count()
print(f"Number of logical CPU cores: {logical_cores}")

num_workers = max(1, logical_cores // 2)
print(f"Number of workers set to: {num_workers}")

def is_gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

gpu_available = is_gpu_available()
print(f"GPU available: {gpu_available}")

print(xgboost.build_info())

ticker_symbol_file = "../ticker-symbol.txt"

make_all_directory()
news_sentiment = NewsSentiment()

# Check if the ticker symbol file exists
if not os.path.isfile(ticker_symbol_file):
    print(f"Ticker symbol file '{ticker_symbol_file}' does not exist.")
else:
    # Read ticker symbols from file
    with open(ticker_symbol_file, 'r') as file:
        ticker_symbols = file.readlines()

    ticker_symbol_list = [ticker_symbol.strip() for ticker_symbol in ticker_symbols]
    end_date = datetime.today()
    start_date = end_date - timedelta(days=3 * 365)

    for ticker_symbol in ticker_symbol_list:
        df, query_search = get_data_from_yahoo_finance(ticker_symbol, start_date, end_date)
        if df is None:
            continue

        df = get_technical_indicator(df)
        df = news_sentiment.get_news_sentiment_score_by_feedparser(df, query_search)
        df = set_target(df)

        split_train_and_test_data_and_save(df, 90, ticker_symbol)

    path = '../data/train'

    ticker_list = []

    if os.path.exists(path):
        ticker_list = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.csv')]

    for ticker_symbol in ticker_list:
        dataframe = pd.read_csv(f"../data/train/{ticker_symbol}.csv")
        X, y_classifier, y_regressor = training_preprocess_data(dataframe)
        xbclassifier_resume_training(X, y_classifier, gpu_available, ticker_symbol, True, True)
        xbregressor_resume_training(X, y_regressor, gpu_available, ticker_symbol, True, True)

    for ticker_symbol in ticker_list:
        dataframe = pd.read_csv(f"../data/test/{ticker_symbol}.csv")
        X = testing_preprocess_data(dataframe)

        sign_correct = 0
        total_predictions = 0
        actual_changes = []
        predicted_changes = []
        predicted_midprice = []

        # Loop through the dataframe starting from row 31 (index 30) to the second-to-last row
        for index, row in X.iloc[30:-1].iterrows():
            row_data = row.to_frame().T.reset_index(drop=True)  # Convert row to DataFrame and reset index

            xbregressor_result = xbregressor_predict(row_data, ticker_symbol)
            predicted_value = xbregressor_result

            predicted_midprice.append(dataframe.iloc[index]['DAILY_MIDPRICE'] + predicted_value)

            actual_change = dataframe.iloc[index]['DAILY_MIDPRICE_CHANGE']
            actual_sign = np.sign(actual_change)

            actual_changes.append(actual_change)
            predicted_changes.append(predicted_value)

            predicted_sign = np.sign(xbregressor_result)

            # Check if the predicted sign matches the actual sign
            if predicted_sign == actual_sign:
                sign_correct += 1
            total_predictions += 1

            print(f"{ticker_symbol} at {dataframe.iloc[index]['Date']} , Actual Price: {actual_change}, xbregressor_result: {xbregressor_result}")

        # Calculate sign accuracy
        sign_accuracy = sign_correct / total_predictions
        print(f"{ticker_symbol} Sign Accuracy: {sign_accuracy}")

        # Calculate RMSE
        rmse = root_mean_squared_error(actual_changes, predicted_changes)
        print(f"{ticker_symbol} RMSE: {rmse}")

        # Predict the last entry
        last_row = X.iloc[-1].to_frame().T.reset_index(drop=True)

        last_xbregressor_result = xbregressor_predict(last_row, ticker_symbol)
        last_predicted_value = last_xbregressor_result

        print(f"{ticker_symbol} at {dataframe.iloc[-1]['Date']} , Predicted Price: {last_predicted_value}")

        # Plotting the results for each ticker
        plt.figure(figsize=(30, 10))

        # Plot actual midprice
        plt.plot(dataframe.iloc[30:-1]['Date'], dataframe.iloc[30:-1]['NEXT_DAY_MIDPRICE'], label='Actual Midprice',
                 color='blue')

        # Plot predicted midprice
        plt.plot(dataframe.iloc[30:-1]['Date'], predicted_midprice, label='Predicted Midprice', color='red',
                 linestyle='--')

        # Add title and labels
        plt.title(f'Actual vs Predicted Midprice for {ticker_symbol}')
        plt.xlabel('Date')
        plt.ylabel('Midprice')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Add legend
        plt.legend()

        # Show the plot
        plt.show()
