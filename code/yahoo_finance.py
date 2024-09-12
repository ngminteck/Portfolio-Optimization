import yfinance as yf


def remove_last_two_words(text):
    words = text.split()
    return ' '.join(words[:-2])


def get_data_from_yahoo_finance(ticker_symbol, start_date, end_date):
    df = None
    query_search = []
    try:
        df = yf.download(ticker_symbol, start=start_date, end=end_date)

        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        query_search.append(ticker_symbol)

        ticker_type = info.get('quoteType', 'Unknown')
        if ticker_type == 'EQUITY':
            company_name = info.get('shortName', '')
            # industry = info.get('industry', '')
            # sector = info.get('sector', '')
            query_search.append(company_name)
            # query_search.append(industry)
            # query_search.append(sector)
        elif ticker_type == 'CURRENCY':
            currency_name = info.get('currency', '')
            query_search.append(currency_name)
        elif ticker_type == 'FUTURE':
            commodity_name = info.get('shortName', '')
            commodity_name = remove_last_two_words(commodity_name)
            query_search.append(commodity_name)
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")

    return df, query_search
