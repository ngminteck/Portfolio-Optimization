import numpy as np
import pandas as pd
from datetime import timedelta
def set_target(df):
    # Create new columns in a separate DataFrame
    new_columns = pd.DataFrame(index=df.index)

    # Shift the 'Close' column by one to get the next day's close price
    new_columns['NEXT_DAY_CLOSEPRICE'] = df['Close'].shift(-1)

    # Calculate the change in close price from one day to the next
    new_columns['DAILY_CLOSEPRICE_CHANGE'] = new_columns['NEXT_DAY_CLOSEPRICE'] - df['Close']

    # Calculate the percentage change in close price
    new_columns['DAILY_CLOSEPRICE_CHANGE_PERCENT'] = (new_columns['DAILY_CLOSEPRICE_CHANGE'] / df['Close'])

    # Determine the direction of the close price change
    new_columns['DAILY_CLOSEPRICE_DIRECTION'] = np.sign(new_columns['DAILY_CLOSEPRICE_CHANGE'])

    # Calculate the daily mid price as the average of the high and low prices
    new_columns['DAILY_MIDPRICE'] = (df['High'] + df['Low']) / 2

    # Shift the 'DAILY_MIDPRICE' column by one to get the next day's mid price
    new_columns['NEXT_DAY_MIDPRICE'] = new_columns['DAILY_MIDPRICE'].shift(-1)

    # Calculate the change in mid price from one day to the next
    new_columns['DAILY_MIDPRICE_CHANGE'] = new_columns['NEXT_DAY_MIDPRICE'] - new_columns['DAILY_MIDPRICE']

    # Calculate the percentage change in mid price
    new_columns['DAILY_MIDPRICE_CHANGE_PERCENT'] = \
                (new_columns['DAILY_MIDPRICE_CHANGE'] / new_columns['DAILY_MIDPRICE'])

    # Determine the direction of the mid price change
    new_columns['DAILY_MIDPRICE_DIRECTION'] = np.sign(new_columns['DAILY_MIDPRICE_CHANGE'])

    # Handle the last row where changes are NaN
    new_columns.at[df.index[-1], 'DAILY_CLOSEPRICE_CHANGE'] = np.nan
    new_columns.at[df.index[-1], 'DAILY_CLOSEPRICE_CHANGE_PERCENT'] = np.nan
    new_columns.at[df.index[-1], 'DAILY_CLOSEPRICE_DIRECTION'] = np.nan
    new_columns.at[df.index[-1], 'DAILY_MIDPRICE_CHANGE'] = np.nan
    new_columns.at[df.index[-1], 'DAILY_MIDPRICE_CHANGE_PERCENT'] = np.nan
    new_columns.at[df.index[-1], 'DAILY_MIDPRICE_DIRECTION'] = np.nan

    # Concatenate the new columns with the original DataFrame
    df = pd.concat([df, new_columns], axis=1)

    return df


def split_train_and_test_data_and_save(df, cutoff_date, ticker_symbol):
    train_df = df[df.index < cutoff_date]
    test_df1 = train_df.tail(30).copy(deep=True)
    test_df2 = df[df.index >= cutoff_date]

    test_df = pd.concat([test_df1, test_df2])

    train_df.to_csv(f'../data/train/{ticker_symbol}.csv')
    test_df.to_csv(f'../data/test/{ticker_symbol}.csv')