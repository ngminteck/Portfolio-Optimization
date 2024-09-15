from yahoo_finance import *
from technical_indicator import *
from news_sentiment_analysis import *
from data_preparation import *

def main_data_collection():
    currencies = ['SGD=X', 'SGDMYR=X', 'GBPSGD=X', 'EURSGD=X', 'SGDJPY=X', 'SGDHKD=X', 'SGDIDR=X', 'SGDCNY=X', 'SGDTHB=X', 'SGDINR=X', 'SGDKRW=X',
                  'AUDSGD=X', 'NZDSGD=X', 'GBPUSD=X', 'JPY=X', 'HKD=X', 'MYR=X', 'INR=X', 'CNY=X', 'PHP=X', 'IDR=X', 'THB=X', 'CHF=X', 'MXN=X',
                  'AUDUSD=X', 'NZDUSD=X', 'KRW=X', 'VND=X', 'CAD=X', 'EURJPY=X', 'GBPJPY=X', 'EURGBP=X', 'EURSEK=X', 'EURCHF=X', 'EURHUF=X', 'EURJPY=X']


    rice_exporter_currencies = ['CNYUSD=X', 'CNYKRW=X', 'CNYJPY=X', 'EGP=X', 'TRY=X', 'PGK=X', 'INR=X', 'IRR=X', 'SAR=X', 'XAF=X', 'BDT=X', 'INRAED=X', 'NGR=X', 'NPR=X',
                  'IDRPHP=X', 'IDRMYR=X', 'IDRSGD=X', 'MMK=X', 'THB=X', 'VND=X', 'VNDPHP=X', 'CNYVND=X', 'IDR=X', 'XOF=X', 'GHS=X', 'VNDMYR=X', 'IQD=X',
                  'THBCNY=X', 'THBZAR=X', 'THBHKD=X', 'CNYMMK=X', 'PHP=X', 'EURMMK=X', 'KHR=X', 'EUR=X', 'MYR=X', 'PKR=X', 'MYRPKR=X', 'KES=X', 'PKRAED=X']

    YEARS = 10

    news_sentiment = NewsSentiment()

    ticker_symbol_list = currencies + rice_exporter_currencies

    end_date = datetime.today()
    start_date = end_date - timedelta(days= YEARS * 365)

    for ticker_symbol in ticker_symbol_list:
        df, query_search = get_data_from_yahoo_finance(ticker_symbol, start_date, end_date)
        if df is None or len(df) == 0:
            continue
        print(f"Fetching {ticker_symbol} from yahoo finance was successfully")
        df = get_technical_indicator(df)
        print(f"Fetching {ticker_symbol} technical indicator was successfully")
        df = news_sentiment.get_news_sentiment_score_by_feedparser(df, query_search)
        print(f"Fetching {ticker_symbol} news sentiment was successfully")
        df = set_target(df)
        print(f"{ticker_symbol} set target successfully")

        cut_off_date = datetime.today() - timedelta(days=3 * 365)
        split_train_and_test_data_and_save(df, cut_off_date, ticker_symbol)

#main_data_collection()