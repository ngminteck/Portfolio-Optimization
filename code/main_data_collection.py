from yahoo_finance import *
from technical_indicator import *
from data_preparation import *

from datetime import datetime
import os
def main_data_collection():

    path = '../data/commodities_historical_data/original'

    commoditieslist = []


    if os.path.exists(path):
        commodities_list = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.csv')]

    print(commodities_list)

    column_name = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    for commodities in commodities_list:
        df = pd.read_csv(f"{path}/{commodities}.csv")
        for column in df.columns:
            if column not in column_name:
                column_name.append(column)

    print(column_name)

    # Second loop to reorder, fill missing columns, and handle 'Settle' column
    for commodities in commodities_list:

        df = pd.read_csv(f"{path}/{commodities}.csv")

        # Copy 'Settle' to 'Close' if 'Settle' exists
        if 'Settle' in df.columns:
            df['Close'] = df['Settle']

        # Copy 'Prev. Day Open Interest' to 'Temp' if 'Prev. Day Open Interest' exists
        if 'Prev. Day Open Interest' in df.columns:
            df['Temp'] = df['Prev. Day Open Interest']


        # Add missing columns and reorder
        for column in column_name:
            if column not in df.columns:
                df[column] = 0

        if 'Temp' in df.columns:
            df['Previous Day Open Interest'] = df['Temp']
            df = df.drop(columns=['Temp'], axis=1)



        df = df.fillna(0)
        # reorder
        df = df[column_name]
        df = df.drop(columns=['Settle'], axis=1)
        df = df.drop(columns=['Prev. Day Open Interest'], axis=1)

        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Set 'Date' column as index
        df.set_index('Date', inplace=True)
        df.index = df.index.tz_localize(None)

        df = get_technical_indicator(df)
        print(f"Fetching {commodities} technical indicator was successfully")
        query_search = []
        query_search.append(commodities)
        df = set_target(df)
        print(f"{commodities} set target successfully")

        cut_off_date = datetime.today() - timedelta(days=3 * 365)
        split_train_and_test_data_and_save(df, cut_off_date, commodities)
        print(f"{commodities} save successfully")

    if 'Date' in column_name:
        column_name.remove('Date')

    if 'Settle' in column_name:
        column_name.remove('Settle')

    if 'Prev. Day Open Interest' in column_name:
        column_name.remove('Prev. Day Open Interest')

    currencies = ['SGD=X', 'SGDMYR=X', 'GBPSGD=X','EURSGD=X', 'SGDJPY=X', 'SGDHKD=X', 'SGDIDR=X', 'SGDCNY=X', 'SGDTHB=X', 'SGDINR=X', 'SGDKRW=X',
                  'AUDSGD=X', 'NZDSGD=X', 'GBPUSD=X', 'JPY=X', 'HKD=X', 'MYR=X', 'INR=X', 'CNY=X', 'PHP=X', 'IDR=X', 'THB=X', 'CHF=X', 'MXN=X',
                  'AUDUSD=X', 'NZDUSD=X', 'KRW=X', 'VND=X', 'CAD=X', 'EURJPY=X', 'GBPJPY=X', 'EURGBP=X', 'EURSEK=X', 'EURCHF=X', 'EURHUF=X', 'EURJPY=X']


    rice_exporter_currencies = ['CNYUSD=X', 'CNYKRW=X', 'CNYJPY=X', 'EGP=X', 'TRY=X', 'PGK=X', 'INR=X', 'IRR=X', 'SAR=X', 'XAF=X', 'BDT=X', 'INRAED=X', 'NGR=X', 'NPR=X',
                  'IDRPHP=X', 'IDRMYR=X', 'IDRSGD=X', 'MMK=X', 'THB=X', 'VND=X', 'VNDPHP=X', 'CNYVND=X', 'IDR=X', 'XOF=X', 'GHS=X', 'VNDMYR=X', 'IQD=X',
                  'THBCNY=X', 'THBZAR=X', 'THBHKD=X', 'CNYMMK=X', 'PHP=X', 'EURMMK=X', 'KHR=X', 'EUR=X', 'MYR=X', 'PKR=X', 'MYRPKR=X', 'KES=X', 'PKRAED=X']

    YEARS = 10

    ticker_symbol_list = currencies + rice_exporter_currencies

    end_date = datetime.today()
    start_date = end_date - timedelta(days= YEARS * 365)

    for ticker_symbol in ticker_symbol_list:

        df, query_search = get_data_from_yahoo_finance(ticker_symbol, start_date, end_date)

        if df is None or len(df) <= 30:
            print(f"{ticker_symbol} from yahoo finance have less than or equal to 30 entry and will be skipped.")
            continue

        for column in column_name:
            if column not in df.columns:
                df[column] = 0

        # missing value all 0
        df = df.fillna(0)
        # reorder
        df = df[column_name]

        print(f"Fetching {ticker_symbol} from yahoo finance was successfully")
        df = get_technical_indicator(df)
        print(f"Fetching {ticker_symbol} technical indicator was successfully")
        df = set_target(df)
        print(f"{ticker_symbol} set target successfully")
        cut_off_date = datetime.today() - timedelta(days=2 * 365)
        split_train_and_test_data_and_save(df, cut_off_date, ticker_symbol)
        print(f"{ticker_symbol} save successfully")

